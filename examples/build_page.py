#!/usr/bin/env python3
"""Fill the examples (and the mixed-line figure) of the website page from out/examples.json,
out/wikisource.json and out/mixed.json, and copy the images next to the page.

    python3 build_page.py --page path/to/tesseract-hebrew.html [--out out/]

The examples compare heb_rivok with Ubuntu's heb, the Hebrew model almost every Tesseract user
has. Each page first shows the lines heb_rivok reads better; a button shows every line, and the
page states how many lines are better, worse and the same.
Replaces the text between <!-- EXAMPLES:BEGIN/END --> and <!-- MIXED:BEGIN/END -->.
"""
import argparse, difflib, html, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLD = str.maketrans({"״": '"', "׳": "'", "־": "-", "–": "-", "—": "-", "“": '"', "”": '"', "„": '"', "’": "'", "‘": "'"})
BASE, OURS = ("ubuntu_heb", "Ubuntu heb"), ("heb_rivok", "heb_rivok")
SHOW = ["bank", "rental", "masa_baarav_1922_p4", "bostenay_1872_p42"]   # tab order
MAX_SHOWN = 5   # lines shown per page: heb_rivok at most 1 error, biggest gap over Ubuntu heb first
NOTES = {
    "invoice": "Text we wrote, set in IBM Plex Sans Hebrew, "
               "damaged like an office scan: 200 dpi, slight tilt, noise, JPEG.",
    "notice": "Text we wrote, set in Bona Nova, damaged like a "
              "worn photocopy: detail lost, then thresholded, with specks.",
    "bank": "Text we wrote, set in Miriam CLM, sent through a "
            "simulated fax: half the vertical resolution, pure black and white.",
    "rental": "Text we wrote, set in Frank Ruehl CLM, damaged "
              "like a worn photocopy.",
    "medical": "Text we wrote, set in Simple CLM, damaged like an office scan.",
    "water": "Text we wrote, set in Nachlieli CLM, damaged like a phone photo.",
    "news": "Text we wrote, set in David CLM, damaged like an office scan.",
}
E = html.escape


def diff_html(ref, out):
    """The model's text with wrong/extra characters marked, missing ones struck through, and
    typographic variants (" for ״, - for ־) marked but not counted."""
    a, b = ref.translate(FOLD), out.translate(FOLD)
    parts = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                parts.append(f'<span class="v" title="typographic variant of {E(ref[i])}">{E(out[j])}</span>'
                             if out[j] != ref[i] else E(out[j]))
        else:
            if i2 > i1 and op in ("replace", "delete"):
                parts.append(f'<del title="missing">{E(ref[i1:i2])}</del>')
            if j2 > j1:
                parts.append(f'<mark title="wrong or extra">{E(out[j1:j2])}</mark>')
    return "".join(parts) or '<span class="empty">(nothing)</span>'


def badge(n):
    return '<span class="n ok">✓</span>' if n == 0 else f'<span class="n">{n} error{"s" if n > 1 else ""}</span>'


def line_html(r, k, scale, shown):
    w, h = r["size"]
    rows = "".join(f'<div class="ro" data-m="{key}"><span class="who">{lab}</span>'
                   f'<p class="txt" dir="rtl" lang="he">{diff_html(r["ref"], r["out"][key])}</p>{badge(r["edits"][key])}</div>'
                   for key, lab in (BASE, OURS))
    cls = "" if shown else ' class="more"'
    return (f'<li{cls}><div class="strip"><img src="tesseract-hebrew/{r["strip"]}" width="{round(w * scale)}" '
            f'height="{round(h * scale)}" alt="Line {k}: {E(r["ref"])}" lang="he" loading="lazy" /></div>{rows}</li>')


def page_panel(p, active):
    scale = p.get("scale", 0.62)
    u = [r["edits"][BASE[0]] for r in p["rows"]]
    o = [r["edits"][OURS[0]] for r in p["rows"]]
    better = sum(b < a for a, b in zip(u, o))
    worse = sum(b > a for a, b in zip(u, o))
    same = len(u) - better - worse
    iw, ih = p["image_size"]
    better_idx = [i for i, (a, b) in enumerate(zip(u, o)) if b < a]
    # heb_rivok at most 1 error first, then the biggest gap over Ubuntu heb
    pick = set(sorted(better_idx, key=lambda i: (o[i] > 1, -(u[i] - o[i]), o[i], i))[:MAX_SHOWN])
    lines = "".join(line_html(r, k, scale, k - 1 in pick) for k, r in enumerate(p["rows"], 1))
    return f'''
        <div class="ex-panel" id="ex-{p["id"]}" role="tabpanel" aria-labelledby="tab-{p["id"]}"{"" if active else " hidden"}>
          <div class="ex-head">
            <a class="ex-page" href="tesseract-hebrew/{p["image"]}" target="_blank" rel="noopener">
              <img src="tesseract-hebrew/{p["image"]}" width="{iw // 4}" height="{ih // 4}" alt="{E(p["title"])}: the whole page" loading="lazy" />
              <span>Whole page ↗</span></a>
            <div>
              <div class="ex-cer">
                <div><span class="k">Ubuntu heb</span><span class="v">{p["cer"][BASE[0]]:.1f}%</span></div>
                <div class="arrow" aria-hidden="true">→</div>
                <div class="ours"><span class="k">heb_rivok</span><span class="v">{p["cer"][OURS[0]]:.1f}%</span></div>
              </div>
              <p class="ex-sub">Characters read wrong, whole page. Of its {len(u)} lines, heb_rivok reads
                {better} better, {worse} worse and {same} the same.</p>
              <p class="table-note">{p.get("note") or NOTES.get(p["id"], "")}</p>
            </div>
          </div>
          <ol class="ex-lines">{lines}</ol>
          <button class="ex-all" aria-expanded="false">Show all {len(u)} lines</button>
        </div>'''


def mixed_figure(m):
    labels = {"heb_rivok_psm13": "heb_rivok, one line", "paddleocr_hebrew": "paddleocr-hebrew"}
    rows = "".join(f'<div class="ro"><span class="who">{lab}</span><p class="txt" dir="rtl" lang="he">'
                   f'{diff_html(m["ref"], m["out"][k])}</p></div>' for k, lab in labels.items())
    return f'''<!-- MIXED:BEGIN -->
        <div class="ex ex-fig"><ol class="ex-lines"><li>
          <div class="strip"><img src="tesseract-hebrew/{m["strip"]}" alt="{E(m["ref"])}" lang="he" /></div>
          <div class="ro"><span class="who">Correct</span><p class="txt" dir="rtl" lang="he">{E(m["ref"])}</p></div>{rows}
        </li></ol></div>
        <!-- MIXED:END -->'''


def replace(s, tag, frag):
    i, j = s.index(f"<!-- {tag}:BEGIN -->"), s.index(f"<!-- {tag}:END -->") + len(f"<!-- {tag}:END -->")
    return s[:i] + frag + s[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True, type=Path)
    ap.add_argument("--out", default=HERE / "out", type=Path, help="where make_examples.py & co. wrote their output")
    a = ap.parse_args()
    OUT = a.out
    pages = {p["id"]: p for p in json.loads((OUT / "examples.json").read_text())["pages"]}
    pages |= {p["id"]: p for p in json.loads((OUT / "wikisource.json").read_text())["pages"]}
    pages = [pages[i] for i in SHOW]
    mixed = json.loads((OUT / "mixed.json").read_text())

    img_dir = a.page.parent / "tesseract-hebrew"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir()
    for f in [p["image"] for p in pages] + [r["strip"] for p in pages for r in p["rows"]] + [mixed["strip"]]:
        shutil.copy2(OUT / f, img_dir / f)

    tabs = "".join(f'<button role="tab" id="tab-{p["id"]}" aria-controls="ex-{p["id"]}" aria-selected="{"true" if i == 0 else "false"}">{E(p["title"])}</button>'
                   for i, p in enumerate(pages))
    frag = f'''<!-- EXAMPLES:BEGIN -->
        <div class="ex">
          <div class="ex-tabs" role="tablist" aria-label="Example documents">{tabs}</div>
          {"".join(page_panel(p, i == 0) for i, p in enumerate(pages))}
          <p class="ex-key"><mark>wrong or extra</mark> <del>missing</del> <span class="v">typographic variant, not counted</span> <span>Spaces before punctuation are not counted either: old print sets one before every comma, modern transcriptions don't.</span></p>
        </div>
        <!-- EXAMPLES:END -->'''
    s = replace(a.page.read_text(), "EXAMPLES", frag)
    s = replace(s, "MIXED", mixed_figure(mixed))
    a.page.write_text(s)
    print(f"wrote {len(pages)} example pages + mixed figure; {len(list(img_dir.iterdir()))} images")


if __name__ == "__main__":
    main()
