#!/usr/bin/env python3
"""Rebuild the real printed training lines of heb_rivok 2.0 from the scans and their Wikisource
transcriptions, and (optionally) check the result against the published line dataset.

Per page of pages.tsv (the training pages and the 23 held-out Seder Tefilot pages):
  1. page image: the scan's page, extracted losslessly with pdfimages when the PDF page holds one
     image, otherwise rendered with pdftoppm at the largest image's resolution (200-600 ppi), grayscale;
  2. line boxes: Tesseract 5.5.0 with heb_rivok 1.0.1 (int8), psm 3, hOCR, one thread
     (or our hOCR files from the dataset, --hocr);
  3. alignment: the page's rendered Wikisource text (the HTML of the revision in pages.tsv; the
     dataset has the HTML we used) is aligned word by word to the OCR, and cut at the OCR's line
     breaks (align.py). A line is kept if both its ends matched and its OCR-vs-text error is <= 25%;
  4. labels: the rules in labels.py (drop lines whose text differs from the print in known ways,
     normalise a few encodings; training/v2/label_changes.tsv lists every changed label);
  5. the crop: the line box, 6 px wider on each side and 15% of its height taller.
Then the Seder Tefilot pages are split as in training: whole pages, shuffled with seed 2, are held
out until they hold >= 300 lines (the vowelled dev set); the rest is the training pool.

Outputs in OUT: pages/BOOK/pNNNN.png, hocr/BOOK/pNNNN.hocr (if OCR was run), align/BOOK/pNNNN.json,
pool/BOOK/pNNNN_lNNN.png + .gt.txt, pool.tsv (book, page, line, label, in the training order: row i
is r{i:06d} in the training lists), dev_tefilot.tsv, dropped.tsv (line and reason).

    python3 -I build_lines.py OUT --books V2_SOURCES/books --page-text DATASET/pages \\
        --unicharset DATASET/models/start/heb_rivok_1.0_avg5.lstm-unicharset \\
        (--hocr DATASET/hocr | --tessdata DIR_WITH_heb_rivok_1.0.1) [--jobs 4] [--check DATASET]

Needs poppler-utils (pdfimages, pdftoppm), Pillow and, without --hocr, tesseract 5.5.0.
"""
import argparse
import collections
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import align as A  # noqa: E402
import labels as L  # noqa: E402

MAX_CER = 0.25
TEFILOT_DEV_LINES, TEFILOT_SEED = 300, 2


# ---------------------------------------------------------------- 1. page images (as extract_pages.py)
def page_images(pdf, n):
    r = subprocess.run(["pdfimages", "-list", "-f", str(n), "-l", str(n), str(pdf)], capture_output=True, text=True)
    rows = []
    for line in r.stdout.splitlines()[2:]:
        f = line.split()
        if len(f) >= 14 and f[2] == "image":
            rows.append({"xppi": int(f[12]) if f[12].isdigit() else 0})
    return rows


def extract_page(pdf, n, out):
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs = page_images(pdf, n)
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        if len(imgs) == 1:
            subprocess.run(["pdfimages", "-png", "-f", str(n), "-l", str(n), str(pdf), f"{td}/x"], check=True,
                           capture_output=True)
            got = sorted(Path(td).glob("x-*.png"))
            if len(got) == 1:
                shutil.move(got[0], out)
                return
        ppi = max([i["xppi"] for i in imgs] or [300]) or 300
        ppi = min(max(ppi, 200), 600)
        subprocess.run(["pdftoppm", "-r", str(ppi), "-gray", "-png", "-singlefile", "-f", str(n), "-l", str(n),
                        str(pdf), f"{td}/r"], check=True, capture_output=True)
        shutil.move(Path(td) / "r.png", out)


# ---------------------------------------------------------------- 2-3. OCR and alignment (as run_align.py)
def do_page(args):
    out, book, n, pdf, text_json, hocr_src, tessdata = args
    png = out / "pages" / book / f"p{n:04d}.png"
    extract_page(pdf, n, png)
    if hocr_src:
        hocr = hocr_src
    else:
        (out / "hocr" / book).mkdir(parents=True, exist_ok=True)
        A.TESSDATA = tessdata
        hocr = A.run_tesseract(png, out / "hocr" / book / f"p{n:04d}", 3)
    page = json.loads(Path(text_json).read_text(encoding="utf-8"))
    body, notes, _ = A.clean_html(page["html"])
    rows = A.align_page(A.read_hocr(Path(hocr)), (body + " " + notes).strip(), "wiki")
    for r in rows:
        r["kept"] = bool(r["gt"]) and r["anchored"] and r["cer_nomarks"] <= MAX_CER
    aj = out / "align" / book / f"p{n:04d}.json"
    aj.parent.mkdir(parents=True, exist_ok=True)
    aj.write_text(json.dumps({"book": book, "page": n, "quality": page["quality"], "revid": page["revid"],
                              "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return book, n


# ---------------------------------------------------------------- 4. labels (as finalize.py + real_lines.py)
def label_of(r, book, quality, ok_chars):
    """(label, None) for a training line, or (None, reason)."""
    if not r["kept"]:
        return None, "not_kept"
    label = r["gt"]
    prob = L.label_problem(label, r["ocr"])
    if prob:
        return None, prob
    if book in L.VOWELLED_ONLY and len(L.NIQ.findall(label)) < 3:
        return None, "unvowelled"
    letters = len(re.findall(r"[א-ת]", label)) or 1
    if (len(L.NIQ.findall(label)) >= 3 and len(L.NIQ.findall(label)) / letters >= 0.4
            and not L.NIQ.search(r["ocr"])):
        label = L.COMB.sub("", label)
    label = L.normalise(label, collections.Counter())
    if L.CANT.search(label):
        return None, "cantillation"
    if "mw-parser-output" in label or re.search(r"[{}]", label):
        return None, "css"
    if not label:
        return None, "empty"
    label = label.replace("׃", ":").replace("‍", "")
    ocr = r["ocr"]
    if quality == 4:
        return None, "validated_page"
    if set(label) - ok_chars:
        return None, "charset"
    if L.plene(label, ocr):
        return None, "plene"
    if len(L.ABBR.findall(ocr)) > len(L.ABBR.findall(label)) and len(label.split()) > len(ocr.split()):
        return None, "abbreviation"
    if (L.NOTE.search(label) and not L.NOTE.search(ocr)) or ("[" in label and "[" not in ocr):
        return None, "editor_note"
    return label, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--books", type=Path, required=True, help="the scans (fetch_v2_sources.py --only books)")
    ap.add_argument("--page-text", type=Path, required=True, help="DATASET/pages: BOOK/pNNNN.json with the HTML")
    ap.add_argument("--unicharset", type=Path, required=True, help="the 1.0 start model's lstm-unicharset")
    ap.add_argument("--pages", type=Path, default=HERE.parent / "pages.tsv")
    ap.add_argument("--hocr", type=Path, help="DATASET/hocr: use our OCR instead of running Tesseract")
    ap.add_argument("--tessdata", type=Path, help="folder with heb_rivok.traineddata 1.0.1 (int8)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--check", type=Path, help="the published dataset folder: compare labels and crops")
    a = ap.parse_args()
    if not a.hocr and not a.tessdata:
        sys.exit("give --hocr (our OCR) or --tessdata (to run Tesseract)")
    head, *rows = [l.split("\t") for l in a.pages.read_text(encoding="utf-8").splitlines()]
    pages = sorted({(r[head.index("book")], int(r[head.index("pdf_page")])) for r in rows})
    todo = []
    for book, n in pages:
        pdf = a.books / f"{book}.pdf"
        hocr = a.hocr / book / f"p{n:04d}.hocr" if a.hocr else None
        todo.append((a.out, book, n, pdf, a.page_text / book / f"p{n:04d}.json", hocr,
                     str(a.tessdata) if a.tessdata else None))
    print(f"{len(todo)} pages", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for k, _ in enumerate(ex.map(do_page, todo, chunksize=2), 1):
            if k % 50 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} pages aligned", flush=True)

    ok_chars = L.allowed_chars(a.unicharset)
    kept, dropped = [], []
    for book, n in pages:
        d = json.loads((a.out / "align" / book / f"p{n:04d}.json").read_text(encoding="utf-8"))
        for r in d["rows"]:
            label, why = label_of(r, book, d["quality"], ok_chars)
            if why == "not_kept":
                continue
            if why:
                dropped.append((book, n, r["line"], why))
            else:
                kept.append((book, n, r, label))
    rng = random.Random(TEFILOT_SEED)
    tef_pages = sorted({n for b, n, _, _ in kept if b == L.TEFILOT})
    rng.shuffle(tef_pages)
    held, count = set(), 0
    for p in tef_pages:
        if count >= TEFILOT_DEV_LINES:
            break
        held.add(p)
        count += sum(1 for b, n, _, _ in kept if b == L.TEFILOT and n == p)
    pool = [k for k in kept if not (k[0] == L.TEFILOT and k[1] in held)]
    dev = [k for k in kept if k[0] == L.TEFILOT and k[1] in held]

    by_page = collections.defaultdict(list)
    for book, n, r, label in pool:
        by_page[(book, n)].append(dict(r, gt=label))
    for (book, n), rs in by_page.items():
        A.crop_lines(a.out / "pages" / book / f"p{n:04d}.png", rs, a.out / "pool" / book, f"p{n:04d}")
    for name, rs in (("pool.tsv", pool), ("dev_tefilot.tsv", dev)):
        with open(a.out / name, "w", encoding="utf-8") as fh:
            for book, n, r, label in rs:
                fh.write(f"{book}\t{n}\t{r['line']}\t{label}\n")
    with open(a.out / "dropped.tsv", "w", encoding="utf-8") as fh:
        for row in dropped:
            fh.write("\t".join(map(str, row)) + "\n")
    print(f"pool {len(pool)} lines, Seder Tefilot dev {len(dev)} lines on {len(held)} pages, "
          f"dropped {dict(collections.Counter(w for *_, w in dropped))}")
    if a.check:
        check(a.out, pool, a.check)


def check(out, pool, ds):
    from PIL import Image, ImageChops
    head, *rows = [l.split("\t") for l in (ds / "lines.tsv").read_text(encoding="utf-8").splitlines()]
    ref = [dict(zip(head, r)) for r in rows if r[head.index("set")] == "pool"]
    mine = [(b, str(n), str(r["line"]), lab) for b, n, r, lab in pool]
    theirs = [(r["book"], r["page"], r["line"], r["label"]) for r in ref]
    print(f"check: {len(mine)} lines rebuilt, {len(theirs)} published;",
          "same lines, same order, same labels" if mine == theirs else
          f"{sum(x != y for x, y in zip(mine, theirs)) + abs(len(mine) - len(theirs))} differ")
    same_bytes = same_pixels = 0
    for r in ref:
        rel = Path(r["image"]).relative_to("pool")
        p, q = out / "pool" / rel, ds / r["image"]
        if not p.exists():
            continue
        if p.read_bytes() == q.read_bytes():
            same_bytes += 1
            same_pixels += 1
        else:
            a, b = Image.open(p), Image.open(q)
            same_pixels += a.size == b.size and a.mode == b.mode and ImageChops.difference(a, b).getbbox() is None
    print(f"check: crops identical byte for byte {same_bytes}/{len(ref)}, pixel for pixel {same_pixels}/{len(ref)}")


if __name__ == "__main__":
    main()
