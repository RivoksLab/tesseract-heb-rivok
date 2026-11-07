#!/usr/bin/env python3
"""HaZefira: real scans of a 19th-century Hebrew newspaper, read line by line and page by page.

Source: OmiLab (Open University of Israel), github.com/omilab/historical_press, file
" hazfirahGT2htr+.zip" (34 Transkribus pages of HaZefira, 1874 and 1887, PAGE XML ground
truth). NO LICENCE IS STATED: the data is downloaded at run time, used for evaluation only,
and must never be committed or redistributed (keep --data-dir outside any repository, or in
the git-ignored eval/data/).

  build   download the zip (pinned commit, sha256 checked) and build, under --data-dir:
            images/<id>.png   each scan with everything outside the transcribed regions painted
                              white, so page OCR is only judged on text that has a reference
            refs/<id>.txt     page reference: line texts, regions in reading order
            lines/<id>_<n>.png + lines.tsv   one crop per transcribed line (polygon bounding
                              box + 6 px, from the unmasked scan): 4,013 crops
            manifest.json
  lines   read every line crop (default psm 13) and print the character error rate
  pages   read every masked page (default psm 3) and print word recall / precision / F1

    python -I hazefira.py build --data-dir data/hazefira [--zip hazfirah.zip]
    python -I hazefira.py lines --data-dir data/hazefira --tessdata ../tessdata --lang heb_rivok
    python -I hazefira.py pages --data-dir data/hazefira --tessdata ../tessdata --lang heb_rivok
    python -I hazefira.py pages --data-dir data/hazefira --lang heb+eng      # the system's heb + eng

Needs: tesseract, Python with Pillow and python-bidi. Scoring: metrics.py.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402

ZIP_URL = ("https://raw.githubusercontent.com/omilab/historical_press/"
           "4908643b90bde64f56704b7375e49e13028fe049/%20hazfirahGT2htr%2B.zip")
ZIP_SHA256 = "775e77227cbd46099487d3294d8cfd449ced7c8b6eeb7865ba41f053fe1b0ea8"
ZIP_ROOT = "120095/hazfirahGT2htr+/"
NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
PAD = 6


# ------------------------------------------------------------------ build
def ro_index(el):
    m = re.search(r"readingOrder \{index:(\d+);", el.get("custom") or "")
    return int(m.group(1)) if m else 10**6


def points(el):
    return [tuple(map(int, p.split(","))) for p in el.find("p:Coords", NS).get("points").split()]


def build(a):
    from PIL import Image, ImageDraw
    out = Path(a.data_dir)
    zpath = Path(a.zip) if a.zip else out / "_download" / "hazfirahGT2htr.zip"
    if not zpath.exists():
        zpath.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {ZIP_URL}", flush=True)
        urllib.request.urlretrieve(ZIP_URL, zpath)
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if sha != ZIP_SHA256:
        sys.exit(f"{zpath}: sha256 {sha}, expected {ZIP_SHA256}")
    for d in ("images", "refs", "lines"):
        (out / d).mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(zpath)
    xmls = sorted(n for n in z.namelist() if n.startswith(ZIP_ROOT + "page/") and n.endswith(".xml"))
    manifest, rows = [], []
    for xml in xmls:
        pid = "hz_" + Path(xml).stem
        t = ET.parse(io.BytesIO(z.read(xml)))
        page = t.find(".//p:Page", NS)
        img = Image.open(io.BytesIO(z.read(ZIP_ROOT + page.get("imageFilename")))).convert("RGB")
        regions = {r.get("id"): r for r in t.findall(".//p:TextRegion", NS)}
        order = [x.get("regionRef") for x in sorted(t.findall(".//p:ReadingOrder//p:RegionRefIndexed", NS),
                                                     key=lambda x: int(x.get("index")))]
        order = [r for r in order if r in regions] + [r for r in regions if r not in order]
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        text_lines = []
        for rid in order:
            reg = regions[rid]
            draw.polygon(points(reg), fill=255)
            for ln in sorted(reg.findall("p:TextLine", NS), key=ro_index):
                u = ln.find("p:TextEquiv/p:Unicode", NS)
                txt = (u.text or "").strip() if u is not None else ""
                if not txt:
                    continue
                text_lines.append(txt)
                xs, ys = zip(*points(ln))
                box = (max(0, min(xs) - PAD), max(0, min(ys) - PAD),
                       min(img.width, max(xs) + PAD), min(img.height, max(ys) + PAD))
                if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                    continue
                crop = out / "lines" / f"{pid}_{len(text_lines):03d}.png"
                if not crop.exists():
                    img.crop(box).save(crop)
                rows.append({"rel_path": f"lines/{crop.name}", "page": pid, "label": txt,
                             "region_type": (re.findall(r"type:([\w-]+)", reg.get("custom") or "") or ["text"])[0]})
        out_img = out / "images" / f"{pid}.png"
        if not out_img.exists():
            white = Image.new("RGB", img.size, "white")
            white.paste(img, mask=mask)
            white.save(out_img)
        ref = out / "refs" / f"{pid}.txt"
        ref.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
        manifest.append({"id": pid, "image": f"images/{pid}.png", "ref_txt": f"refs/{pid}.txt",
                         "lines": len(text_lines), "regions": len(regions), "size": list(img.size)})
        print(f"  {pid}: {len(text_lines)} lines", flush=True)
    json.dump(manifest, open(out / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(out / "lines.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rel_path", "page", "label", "region_type"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(manifest)} pages, {sum(m['lines'] for m in manifest)} reference lines, {len(rows)} line crops")


# ------------------------------------------------------------------ OCR
def tesseract(path, a, psm):
    cmd = ["tesseract", str(path), "stdout", "-l", a.lang, "--psm", str(psm)]
    if a.tessdata:
        cmd[1:1] = ["--tessdata-dir", str(a.tessdata)]
    p = subprocess.run(cmd, capture_output=True, env=dict(os.environ, OMP_THREAD_LIMIT="1"))
    err = p.stderr.decode("utf-8", "replace")
    if p.returncode and ("Error opening data file" in err or "Failed loading language" in err):
        sys.exit(f"tesseract cannot load the model: {' '.join(cmd)}\n{err[-1000:]}")
    return p.stdout.decode("utf-8", "replace")


def run_all(items, fn, jobs, out_file):
    """Run fn over items in parallel; keep results in out_file (JSONL) and reuse them on a rerun."""
    done = {}
    if out_file and Path(out_file).exists():
        done = {r["key"]: r for r in map(json.loads, open(out_file, encoding="utf-8"))}
    todo = [k for k in items if k not in done]
    f = open(out_file, "a", encoding="utf-8") if out_file else None

    def one(k):
        t0 = time.time()
        return {"key": k, "pred": fn(k), "sec": round(time.time() - t0, 3)}
    with ThreadPoolExecutor(jobs) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            done[r["key"]] = r
            if f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}", file=sys.stderr, flush=True)
    if f:
        f.close()
    return done


def data_dir(a):
    d = Path(a.data_dir)
    if not (d / "lines.tsv").exists() or not (d / "manifest.json").exists():
        sys.exit(f"{d} has no HaZefira set; run: python -I hazefira.py build --data-dir {d}")
    return d


def lines(a):
    d = data_dir(a)
    rows = list(csv.DictReader(open(d / "lines.tsv", encoding="utf-8"), delimiter="\t"))
    # lines.tsv paths may carry a prefix (pagesets/hazefira/lines/...): only the file name counts
    path = {r["rel_path"]: d / "lines" / Path(r["rel_path"]).name for r in rows}
    res = run_all([r["rel_path"] for r in rows], lambda k: " ".join(tesseract(path[k], a, a.psm).split()), a.jobs, a.out)
    e = fo = c = et = ct = 0
    for r in rows:
        g, p = M.norm(r["label"]), M.norm(res[r["rel_path"]]["pred"])
        dd = M.dist_display(p, g)
        e += dd
        fo += M.dist_display(p.translate(M.FOLD), g.translate(M.FOLD))
        c += max(len(g), 1)
        if r["region_type"] == "text":
            et += dd
            ct += max(len(g), 1)
    print(f"HaZefira lines, {a.lang} (psm {a.psm}): {len(rows)} lines")
    print(f"  CER {100 * e / c:.2f}%   folded (׳ ״ ־ – as ' \" - -) {100 * fo / c:.2f}%   "
          f"text regions only {100 * et / max(1, ct):.2f}%")


def pages(a):
    d = data_dir(a)
    manifest = json.load(open(d / "manifest.json", encoding="utf-8"))
    img = {e["id"]: d / "images" / Path(e["image"]).name for e in manifest}
    res = run_all([e["id"] for e in manifest], lambda k: tesseract(img[k], a, a.psm), a.jobs, a.out)
    h = nr = np_ = nh = nn = 0
    for e in manifest:
        ref = collections.Counter(M.tokens((d / "refs" / Path(e["ref_txt"]).name).read_text(encoding="utf-8")))
        pred = collections.Counter(M.tokens(res[e["id"]]["pred"]))
        nref = collections.Counter({w: n for w, n in ref.items() if re.search(r"\d", w)})
        h += sum((ref & pred).values()); nr += sum(ref.values()); np_ += sum(pred.values())
        nh += sum((nref & pred).values()); nn += sum(nref.values())
    rec, prec = h / nr, h / max(1, np_)
    print(f"HaZefira pages, {a.lang} (psm {a.psm}): {len(manifest)} pages, {nr} reference words")
    print(f"  recall {100 * rec:.1f}%   precision {100 * prec:.1f}%   F1 {100 * 2 * rec * prec / (rec + prec):.1f}%   "
          f"numbers found {100 * nh / max(1, nn):.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build", "lines", "pages"])
    ap.add_argument("--data-dir", required=True, help="the built set (see build)")
    ap.add_argument("--zip", help="build: use this local copy of the zip instead of downloading")
    ap.add_argument("--tessdata", help="folder holding LANG.traineddata (default: Tesseract's own)")
    ap.add_argument("--lang", help="model name(s) for -l, e.g. heb_rivok, heb, heb+eng")
    ap.add_argument("--psm", type=int, help="page segmentation mode (lines: 13, pages: 3)")
    ap.add_argument("--jobs", type=int, default=2, help="images read in parallel (one thread each)")
    ap.add_argument("--out", help="keep predictions in this JSONL file (a rerun reuses them)")
    a = ap.parse_args()
    if a.cmd == "build":
        return build(a)
    if not a.lang:
        ap.error("--lang is required")
    if a.psm is None:
        a.psm = 13 if a.cmd == "lines" else 3
    return lines(a) if a.cmd == "lines" else pages(a)


if __name__ == "__main__":
    main()
