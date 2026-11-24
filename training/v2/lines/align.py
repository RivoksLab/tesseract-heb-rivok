#!/usr/bin/env python3
"""Align a Wikisource Page's text to Tesseract's lines on its scan; write tesstrain line pairs.

Wikisource Page text is paragraph-based (no printed line breaks) and lightly edited (added
punctuation, modernised quotes, a few corrected or mistyped words). The printed line breaks come
from OCR instead:

1. Clean the wikitext (clean_wikitext): drop <noinclude>, <section>, bidi controls; unwrap
   templates, links, bold/italic and HTML tags; move <ref> footnotes to the end (they print at
   the foot of the page).
2. OCR the scan with Tesseract (hOCR) for line boxes and a rough reading of each line.
3. Align the OCR word stream to the wikitext word stream (Needleman-Wunsch over words, cost =
   normalised edit distance of the words' letters), then cut the wikitext stream at the OCR line
   breaks. Wikitext words that fall in a gap at a line break go with the line before them only if
   they are punctuation; a gap of real words at a line break makes both lines "unanchored".
4. Keep a line only if it is anchored (its first and last OCR words matched wikitext words) and
   the CER between the OCR reading and the wikitext line is at most --max-cer.

Punctuation (--punct): "wiki" keeps Wikisource's punctuation; "ocr" takes the printed
punctuation from the OCR reading wherever the two words have the same letters, and drops
wikitext tokens that are punctuation only and matched nothing (editorial additions).

    python3 align.py PAGE.png WIKITEXT.txt OUTDIR [--hocr PAGE.hocr] [--max-cer 0.2] [--punct ocr]
"""
import argparse
import html
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

BIDI = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩﻿"))
# combining marks only (niqqud, cantillation, dagesh, dots); NOT maqaf 05BE, paseq 05C0, sof pasuq 05C3,
# nun hafukha 05C6, which are printed punctuation
MARKS = re.compile(r"[\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7]")
LETTERS = re.compile(r"[א-ת0-9A-Za-z]")
TESSDATA = "tessdata"   # a folder with heb_rivok 1.0.1 (fast, int8); set by build_lines.py --tessdata


# ---------------------------------------------------------------- rendered HTML (preferred)
class _PageText:
    """Collect the visible text of a ProofreadPage action=parse result.

    Keeps div.pagetext (header/footer and the quality banner are dropped); removes footnote markers
    (sup.reference) and edit links; collects ol.references items as footnotes (printed at the
    foot of the page). Block elements and <br> become spaces.
    """

    SKIP = ("prp-page-qualityheader", "mw-editsection", "reference", "mw-cite-backlink", "noprint", "error")
    BLOCK = {"p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "center", "table", "dd", "dt"}

    def __init__(self, src: str):
        from html.parser import HTMLParser

        self.body, self.notes = [], []
        stack = []  # (tag, skip?, in_pagetext?, in_refs?)
        outer = self

        class P(HTMLParser):
            def handle_starttag(self, tag, attrs):
                cls = dict(attrs).get("class") or ""
                parent = stack[-1] if stack else (None, False, False, False)
                skip = parent[1] or any(c in cls.split() for c in outer.SKIP) or tag in ("style", "script")
                in_pt = parent[2] or "pagetext" in cls.split()
                in_refs = parent[3] or ("references" in cls.split() and tag == "ol")
                if tag in ("br", "img", "hr", "meta", "link", "wbr", "input"):
                    if tag == "br" and in_pt and not skip:
                        (outer.notes if in_refs else outer.body).append(" ")
                    return
                stack.append((tag, skip, in_pt, in_refs))
                if tag in outer.BLOCK and in_pt and not skip:
                    (outer.notes if in_refs else outer.body).append(" ")

            def handle_endtag(self, tag):
                # void elements (<br />) are never pushed; only pop when the tag is open
                if tag in ("br", "img", "hr", "meta", "link", "wbr", "input") or not any(t[0] == tag for t in stack):
                    return
                while stack:
                    t = stack.pop()
                    if t[0] == tag:
                        break
                if tag in outer.BLOCK:
                    outer.body.append(" ")

            def handle_data(self, data):
                if not stack:
                    return
                _, skip, in_pt, in_refs = stack[-1]
                if skip:
                    return
                if in_refs:
                    outer.notes.append(data)
                elif in_pt:
                    outer.body.append(data)

        p = P(convert_charrefs=True)
        p.feed(src)
        p.close()


def clean_html(src: str):
    """Return (body text, footnotes text, notes) from an action=parse HTML of a Page."""
    t = _PageText(src)

    def tidy(s):
        # paseq: Wikisource types it as "|"; U+034F (grapheme joiner, Wikisource's Jerusalem convention) is invisible
        s = s.translate(BIDI).replace("\u00a0", " ").replace("|", " \u05c0 ").replace("\u034f", "")
        s = re.sub(r"(\S)\s+([,.:;])(?=[א-ת])", r"\1\2 ", s)
        return re.sub(r"\s+", " ", s).strip()

    return tidy("".join(t.body)), tidy("".join(t.notes)), ["html"]


# ---------------------------------------------------------------- wikitext
def _unwrap_templates(s: str) -> str:
    """Replace innermost {{...}} until none are left: keep the last positional argument."""
    pat = re.compile(r"\{\{([^{}]*)\}\}")
    for _ in range(50):
        new = pat.sub(_template_text, s)
        if new == s:
            return new
        s = new
    return s


def _template_text(m: re.Match) -> str:
    parts = m.group(1).split("|")
    name = parts[0].strip()
    args = [p for p in parts[1:] if "=" not in p.split("[[")[0]]
    if name in ("ש", "שורה", "br"):  # forced line break
        return " "
    return args[-1] if args else ""


def clean_wikitext(wt: str):
    """Return (body text, footnotes text, notes about markup seen)."""
    seen = set()
    wt = wt.translate(BIDI)
    wt = re.sub(r"<noinclude>.*?</noinclude>", " ", wt, flags=re.S)
    if "<section" in wt:
        seen.add("section")
    wt = re.sub(r"<section[^>]*/>", " ", wt)
    refs = re.findall(r"<ref[^>/]*>(.*?)</ref>", wt, flags=re.S)
    if refs:
        seen.add("ref")
    wt = re.sub(r"<ref[^>/]*>.*?</ref>|<ref[^>]*/>", " ", wt, flags=re.S)
    wt = re.sub(r"<references\s*/>", " ", wt)

    def unwrap(t):
        if "{{" in t:
            seen.update(re.findall(r"\{\{\s*([^|}\n]{1,25})", t))
        t = _unwrap_templates(t)
        t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
        t = re.sub(r"'''?", "", t)
        t = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", t, flags=re.M)
        t = re.sub(r"<br\s*/?>", " ", t)
        t = re.sub(r"</?[a-zA-Z][^>]*>", " ", t)
        return html.unescape(t).replace(" ", " ")

    def fix_bidi_punct(t):
        # a comma/period/colon left at the start of the next Hebrew word by bidi embeddings ("ודונג ,אם")
        return re.sub(r"(\S)\s+([,.:;])(?=[\u05D0-\u05EA])", r"\1\2 ", t)

    return fix_bidi_punct(unwrap(wt)), " ".join(fix_bidi_punct(unwrap(r)) for r in refs), sorted(seen)


# ---------------------------------------------------------------- tokens and costs
def key(w: str) -> str:
    """Letters and digits only (no marks, no punctuation): what matching compares."""
    return "".join(LETTERS.findall(MARKS.sub("", unicodedata.normalize("NFC", w))))


def lev(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    return lev(hyp, ref) / max(1, len(ref))


def sub_cost(a: str, b: str) -> float:
    ka, kb = key(a), key(b)
    if not ka and not kb:
        return 0.0 if a == b else 0.3
    if not ka or not kb:
        return 1.5
    return lev(ka, kb) / max(len(ka), len(kb))


GAP = 0.75


def align_words(ocr, wiki):
    """Needleman-Wunsch; returns list of (i or None, j or None) pairs in order."""
    n, m = len(ocr), len(wiki)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    B = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0], B[i][0] = i * GAP, 1
    for j in range(1, m + 1):
        D[0][j], B[0][j] = j * GAP, 2
    for i in range(1, n + 1):
        oi = ocr[i - 1]
        Di, Dp = D[i], D[i - 1]
        for j in range(1, m + 1):
            best, b = Dp[j - 1] + sub_cost(oi, wiki[j - 1]), 0
            if Dp[j] + GAP < best:
                best, b = Dp[j] + GAP, 1
            if Di[j - 1] + GAP < best:
                best, b = Di[j - 1] + GAP, 2
            Di[j], B[i][j] = best, b
    out, i, j = [], n, m
    while i or j:
        b = B[i][j]
        if i and j and b == 0:
            out.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif i and (b == 1 or not j):
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    return out[::-1]


# ---------------------------------------------------------------- hOCR
def read_hocr(path: Path):
    """[(bbox, [words])] per text line, in Tesseract's reading order."""
    s = path.read_text(encoding="utf-8")
    lines = []
    for m in re.finditer(r"<span class='ocr_(?:line|header|textfloat|caption)' id='[^']*' title=\"bbox (\d+) (\d+) (\d+) (\d+)[^\"]*\">(.*?)</span>\s*(?=<span class='ocr_(?:line|header|textfloat|caption)'|</p>)", s, flags=re.S):
        bbox = tuple(map(int, m.group(1, 2, 3, 4)))
        words = [html.unescape(re.sub(r"<[^>]+>", "", w)).strip()
                 for w in re.findall(r"<span class='ocrx_word'[^>]*>(.*?)</span>", m.group(5), flags=re.S)]
        words = tidy_ocr_words([w for w in words if w])
        if words:
            lines.append((bbox, words))
    return lines


NOISE = re.compile(r"^[_\u2022\u00b7|~^`\u2014\u2013-]{0,1}[_\u2022\u00b7|~^`]+[_\u2022\u00b7|~^`\u2014\u2013-]*$")
DASHES = set("\u2013\u2014\u2012-")
STANDALONE = DASHES | {"\u05c0"}  # dashes and paseq are printed as separate tokens
OPENERS = set('"\u201e\u201c(\u00ab[\u05f4\'')


def tidy_ocr_words(words):
    """Drop pure noise tokens, strip '_' at word edges, glue lone punctuation to its word."""
    out = []
    for w in words:
        w = w.strip("_\u2022\u00b7|~^`")
        if not w or NOISE.match(w):
            continue
        if not key(w) and out and not (set(w) <= OPENERS) and not (set(w) <= STANDALONE):
            out[-1] += w
            continue
        out.append(w)
    # a lone opener glues to the next word
    res = []
    for k, w in enumerate(out):
        if res and not key(res[-1]) and set(res[-1]) <= OPENERS:
            res[-1] += w
        else:
            res.append(w)
    return res


def run_tesseract(png: Path, outbase: Path, psm: int):
    subprocess.run(["tesseract", str(png), str(outbase), "--tessdata-dir", TESSDATA, "-l", "heb_rivok",
                    "--psm", str(psm), "-c", "tessedit_create_hocr=1"], check=True,
                   capture_output=True, env={"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin"})
    return outbase.with_suffix(".hocr")


# ---------------------------------------------------------------- main step
JOINERS = set("-\u05be\u2013\u2014\u2012")  # hyphen, maqaf, en/em/figure dash
HYPHENISH = set("-\u05be\u2013\u2014\u2012=_\u05f4\u2e17\"\u2010\u2011")  # what OCR reads for a printed line-end hyphen
LINE_HYPHEN = "\u05be"  # canonical printed line-break hyphen when Wikisource dropped it


def _trailing_nonletters(w: str) -> str:
    """The run of characters after the last letter/digit of w, combining marks excluded."""
    out = []
    for ch in reversed(w):
        if LETTERS.match(ch):
            break
        if not MARKS.match(ch):
            out.append(ch)
    return "".join(reversed(out))


def _split_at_letter(w: str, k: int):
    """Split w before its k-th kept letter; combining marks and joiners stay with the first part."""
    pos = [i for i, ch in enumerate(w) if LETTERS.match(ch)]
    return w[:pos[k]], w[pos[k]:]


def _joiner_positions(w: str):
    """Letter indices k (as in _split_at_letter) that a joiner directly precedes."""
    pos = [i for i, ch in enumerate(w) if LETTERS.match(ch)]
    return {k for k in range(1, len(pos)) if any(ch in JOINERS for ch in w[pos[k - 1] + 1:pos[k]])}


def split_line_end_words(lines, ocr_words, line_of, wiki, pairs):
    """Words that the print breaks across a line end ("קזלה-" / "מונפרטה") but Wikisource joins.

    At each line boundary, A = last lettered OCR word of the line, B = first lettered OCR word of the
    next. If one of them is aligned to a wiki token W whose letters match key(A)+key(B) much better than
    either fragment alone (and the other fragment matched nothing well), W is split at the best letter
    position: W1 goes to the line above, W2 to the line below. A joiner already in W stays with W1;
    if W has none but the OCR saw a hyphen at the end of A, a line hyphen (maqaf) is added.
    Borderline matches mark both lines as unsure (they are then not kept).
    Returns (new wiki list, {line: "split"}, unsure line set).
    """
    o2w = {i: j for i, j in pairs if i is not None and j is not None}
    lettered = {}
    for i, li in enumerate(line_of):
        if key(ocr_words[i]):
            lettered.setdefault(li, []).append(i)
    splits, touched, unsure = {}, {}, set()
    for li in range(len(lines) - 1):
        if li not in lettered or li + 1 not in lettered:
            continue
        a, b = lettered[li][-1], lettered[li + 1][0]
        ka, kb = key(ocr_words[a]), key(ocr_words[b])
        for j, other in ((o2w.get(a), b), (o2w.get(b), a)):
            if j is None or j in splits:
                continue
            jo = o2w.get(other)
            if jo is not None and sub_cost(ocr_words[other], wiki[jo]) <= 0.5:
                continue  # the other fragment is a word of its own
            kw = key(wiki[j])
            if len(kw) < 3 or len(kw) < max(len(ka), len(kb)) + 2:
                continue
            join = lev(ka + kb, kw) / len(kw)
            single = min(lev(ka, kw), lev(kb, kw)) / len(kw)
            if join > 0.5 or join > single - 0.2:
                continue
            if join > 0.34:
                unsure.update((li, li + 1))
                break
            costs = [(lev(ka, kw[:c]) + lev(kb, kw[c:]), c) for c in range(1, len(kw))]
            best = min(x for x, _ in costs)
            # a joiner inside W (maqaf, hyphen, dash) is where the print breaks it: prefer it when it is
            # (nearly) as good, e.g. "שלוף־הנעל" with the OCR reading the line-end maqaf as "ה"
            jpos = _joiner_positions(wiki[j])
            near = [c for x, c in costs if c in jpos and x <= best + 1]
            k = near[0] if near else min(costs, key=lambda t: (t[0], abs(t[1] - len(ka)) + abs(len(kw) - t[1] - len(kb))))[1]
            w1, w2 = _split_at_letter(wiki[j], k)
            toks = [w1]
            if w1 and w1[-1] in JOINERS:
                nxt = ocr_words[a + 1] if a + 1 < len(ocr_words) and line_of[a + 1] == li else ""
                if nxt and set(nxt) <= JOINERS and len(w1) > 1:
                    toks = [w1[:-1], w1[-1]]  # printed "שילר —": the dash stands apart
            elif set(_trailing_nonletters(ocr_words[a])) & HYPHENISH:
                toks = [w1 + LINE_HYPHEN]
            splits[j] = toks + [w2]
            touched[li] = touched[li + 1] = "split"
            break
    if not splits:
        return wiki, {}, unsure
    new = []
    for j, w in enumerate(wiki):
        new += splits.get(j, [w])
    return new, touched, unsure


def align_page(lines, wiki_text, punct="ocr"):
    ocr_words, line_of = [], []
    for li, (_, ws) in enumerate(lines):
        ocr_words += ws
        line_of += [li] * len(ws)
    wiki = tidy_ocr_words(wiki_text.split())  # same gluing of lone punctuation as for OCR
    pairs = align_words(ocr_words, wiki)
    wiki2, touched, unsure = split_line_end_words(lines, ocr_words, line_of, wiki, pairs)
    if touched:
        wiki = wiki2
        pairs = align_words(ocr_words, wiki)

    # assign wiki tokens to lines
    per_line = [[] for _ in lines]          # (wiki token, matched ocr word or None)
    anchored_first = [False] * len(lines)
    anchored_last = [False] * len(lines)
    boundary_words = [0] * len(lines)       # real words in a gap at the line's end
    last_line = None
    pending = []                            # wiki tokens in a gap since the last matched word
    for k, (i, j) in enumerate(pairs):
        if i is not None and j is not None:
            li = line_of[i]
            if pending:
                if last_line is not None and last_line != li:
                    # gap at a line break: punctuation goes to the previous line; words make both lines unanchored
                    for w in pending:
                        if key(w):
                            boundary_words[last_line] += 1
                            boundary_words[li] += 1
                        per_line[last_line].append((w, None))
                else:
                    per_line[li if last_line is None else last_line] += [(w, None) for w in pending]
                pending = []
            good = sub_cost(ocr_words[i], wiki[j]) <= 0.5
            if not any(o is not None and key(o) for _, o in per_line[li]):
                anchored_first[li] = good
            per_line[li].append((wiki[j], ocr_words[i]))
            anchored_last[li] = good
            last_line = li
        elif j is not None:
            pending.append(wiki[j])
        else:  # OCR word with no wiki word (noise, running head, or missing text)
            li = line_of[i]
            if key(ocr_words[i]):
                if not any(o is not None and key(o) for _, o in per_line[li]):
                    anchored_first[li] = False
                anchored_last[li] = False
                per_line[li].append((None, ocr_words[i]))
            last_line = li
    out = []
    for li, ((bbox, ws), toks) in enumerate(zip(lines, per_line)):
        words = []
        for w, o in toks:
            if w is None:
                continue  # OCR-only word: no text for it
            if o is not None and key(o) and key(w) and sub_cost(o, w) <= 0.5:
                if punct == "ocr":
                    w = _merge_punct(w, o)
                elif punct == "hybrid":
                    w = _hybrid_punct(w, o)
            elif o is None and not key(w) and punct == "ocr" and not (set(w) <= DASHES):
                continue  # editorial punctuation that matched nothing (dashes are kept: rarely editorial)
            words.append(w)
        gt = " ".join(words)
        ocr_line = " ".join(ws)
        out.append({"line": li, "bbox": bbox, "ocr": ocr_line, "gt": gt,
                    "anchored": anchored_first[li] and anchored_last[li] and boundary_words[li] == 0 and li not in unsure,
                    "split_word": touched.get(li, "unsure" if li in unsure else ""),
                    "cer": round(cer(ocr_line, gt), 4) if gt else 1.0,
                    # filter on letters and punctuation only: niqqud in the label is proofread, and the
                    # OCR's niqqud errors must not reject vowelled lines
                    "cer_nomarks": round(cer(MARKS.sub("", ocr_line), MARKS.sub("", gt)), 4) if gt else 1.0})
    return out


TRAIL = re.compile(r"[^\u05D0-\u05EA0-9A-Za-z\u0591-\u05C7]*$")
LEAD = re.compile(r"^[^\u05D0-\u05EA0-9A-Za-z]*")


def _hybrid_punct(wiki_w: str, ocr_w: str) -> str:
    """Wikisource punctuation where it has some (proofread); OCR's trailing mark only where it has none."""
    if TRAIL.search(wiki_w).group(0):
        return wiki_w
    t = TRAIL.search(ocr_w).group(0)
    return wiki_w + t if t and set(t) <= set('.,:;?!"\u05f4\u201d\'') else wiki_w


def _merge_punct(wiki_w: str, ocr_w: str) -> str:
    """Keep the wikitext's letters and marks, take leading/trailing punctuation from the OCR word."""
    lead_o = re.match(r"^[^א-ת0-9A-Za-z]*", ocr_w).group(0)
    trail_o = re.search(r"[^א-ת0-9A-Za-z֑-ׇ]*$", ocr_w).group(0)
    core = re.sub(r"^[^א-ת0-9A-Za-z]*|[^א-ת0-9A-Za-z֑-ׇ]*$", "", wiki_w)
    return lead_o + core + trail_o


def crop_lines(png: Path, rows, outdir: Path, stem: str, pad=6):
    from PIL import Image
    im = Image.open(png).convert("L")
    outdir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        x0, y0, x1, y1 = r["bbox"]
        h = y1 - y0
        box = (max(0, x0 - pad), max(0, y0 - int(0.15 * h)), min(im.width, x1 + pad), min(im.height, y1 + int(0.15 * h)))
        name = f"{stem}_l{r['line']:03d}"
        im.crop(box).save(outdir / f"{name}.png")
        (outdir / f"{name}.gt.txt").write_text(r["gt"] + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png", type=Path)
    ap.add_argument("wikitext", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--hocr", type=Path)
    ap.add_argument("--psm", type=int, default=4)
    ap.add_argument("--max-cer", type=float, default=0.25)
    ap.add_argument("--punct", choices=("ocr", "wiki", "hybrid"), default="wiki")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    hocr = a.hocr or run_tesseract(a.png, a.outdir / a.png.stem, a.psm)
    body, notes, seen = clean_wikitext(a.wikitext.read_text(encoding="utf-8"))
    rows = align_page(read_hocr(hocr), body + " " + notes, a.punct)
    keep = [r for r in rows if r["anchored"] and r["cer"] <= a.max_cer]
    crop_lines(a.png, keep, a.outdir / "lines", a.png.parent.name or a.png.stem)
    json.dump({"markup": seen, "rows": rows}, open(a.outdir / "align.json", "w"), ensure_ascii=False, indent=1)
    print(f"{len(rows)} OCR lines, kept {len(keep)} (anchored and CER <= {a.max_cer}); markup: {seen}", file=sys.stderr)


if __name__ == "__main__":
    main()
