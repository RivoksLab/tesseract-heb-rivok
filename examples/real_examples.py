#!/usr/bin/env python3
"""Real-scan examples: public-domain printed pages from Hebrew Wikisource / Wikimedia Commons,
read as whole pages (default page mode, psm 3) by each model and scored line by line against the
validated Wikisource transcription (CC BY-SA 4.0), which was split into the printed lines.

    python -I real_examples.py --pages PAGES_DIR [--stock STOCK_DIR] [--out out/]

PAGES_DIR/<page id>/ holds page.png (the scan), transcription.txt (one printed line per line)
and source.json (where both came from: Wikisource page and Commons file URLs, revision,
licences). These inputs are not part of this repository. Models: as in make_examples.py.

Lines are found once, from Ubuntu heb's own layout (Tesseract's layout analysis does not depend
on the model), and matched to the transcription in order. Each model's lines go to the nearest
found line; text a model reads where the transcription has none (scan borders, specks) counts
as extra errors for that model.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))      # python -I does not add the script's folder
from make_examples import FOLD, add_model_args, build_models, lev, norm  # noqa: E402

PAGES = {
    "bostenay_1872_p42": ("A novel, printed in Vilna in 1872",
                          "Bostenay (Marcus Lehmann, Hebrew by S. Y. Fuenn), page 42, Vilna 1872"),
    "masa_baarav_1922_p4": ("A travel book, printed in Warsaw in 1922",
                            "Masa ba-Arav (Samuel Romanelli), page ד, Warsaw 1922"),
}


def tess_lines(png, model):
    key, _, tessdir, lang = model
    cmd = ["tesseract", str(png), "stdout", "--psm", "3", "-l", lang, "-c", "tessedit_create_tsv=1"]
    if tessdir:
        cmd[1:1] = ["--tessdata-dir", str(tessdir)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_THREAD_LIMIT="1")).stdout
    lines = {}
    for row in out.splitlines()[1:]:
        f = row.split("\t")
        if len(f) < 12 or f[0] != "5" or not f[11].strip():
            continue
        k = (int(f[2]), int(f[3]), int(f[4]))
        l, t, w, h = map(int, f[6:10])
        d = lines.setdefault(k, {"words": [], "box": [l, t, l + w, t + h]})
        d["words"].append(f[11])
        b = d["box"]
        d["box"] = [min(b[0], l), min(b[1], t), max(b[2], l + w), max(b[3], t + h)]
    res = [{"text": norm(" ".join(v["words"])), "box": v["box"]} for v in lines.values()]
    return sorted(res, key=lambda r: (r["box"][1] + r["box"][3]) / 2)


def align(refs, segs):
    """Monotonic alignment of reference lines to found lines, maximising text similarity."""
    def sim(a, b):
        a, b = a.translate(FOLD), b.translate(FOLD)
        return max(0.0, 1 - lev(a, b) / max(len(a), 1))
    m, n = len(refs), len(segs)
    S = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            S[i][j] = max(S[i - 1][j], S[i][j - 1], S[i - 1][j - 1] + sim(refs[i - 1], segs[j - 1]["text"]))
    pairs, i, j = {}, m, n
    while i and j:
        if S[i][j] == S[i - 1][j]:
            i -= 1
        elif S[i][j] == S[i][j - 1]:
            j -= 1
        else:
            pairs[j - 1] = i - 1
            i, j = i - 1, j - 1
    return pairs     # found-line index -> reference-line index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True, type=Path)
    ap.add_argument("--out", default=Path(__file__).resolve().parent / "out", type=Path)
    add_model_args(ap)
    a = ap.parse_args()
    MODELS = build_models(a)
    a.out.mkdir(parents=True, exist_ok=True)
    result = {"pages": []}
    for pid, (title, cite) in PAGES.items():
        d = a.pages / pid
        src = json.loads((d / "source.json").read_text())
        refs = [norm(x) for x in (d / "transcription.txt").read_text().splitlines() if x.strip()]
        png = d / "page.png"
        img = Image.open(png).convert("L")
        outs = {m[0]: tess_lines(png, m) for m in MODELS}
        segs = outs["ubuntu_heb"]
        pairs = align(refs, segs)
        centres = [(s["box"][1] + s["box"][3]) / 2 for s in segs]
        rows = [{"ref": r, "out": {}, "edits": {}} for r in refs]
        extra = {}
        for key, lines in outs.items():
            got = [[] for _ in refs]
            extra[key] = 0
            for ln in lines:
                y = (ln["box"][1] + ln["box"][3]) / 2
                j = min(range(len(segs)), key=lambda k: abs(centres[k] - y))
                if j in pairs:
                    got[pairs[j]].append(ln["text"])
                else:
                    extra[key] += len(ln["text"])
            for row, parts in zip(rows, got):
                hyp = norm(" ".join(parts))
                row["out"][key] = hyp
                row["edits"][key] = lev(row["ref"].translate(FOLD), hyp.translate(FOLD))
        for j, i in pairs.items():   # line images from the found layout
            l, t, r, b = segs[j]["box"]
            strip = img.crop((max(0, l - 14), max(0, t - 10), min(img.width, r + 14), min(img.height, b + 10)))
            name = f"{pid}_L{i + 1}.jpg"
            strip.convert("RGB").save(a.out / name, quality=85)
            rows[i]["strip"], rows[i]["size"] = name, strip.size
        web = img.copy()
        web.thumbnail((1000, 1600))
        web.convert("RGB").save(a.out / f"{pid}.jpg", quality=80)
        total = sum(len(r["ref"]) for r in rows)
        cer = {k: round(100 * (sum(r["edits"][k] for r in rows) + extra[k]) / total, 2) for k in outs}
        rows = [r for r in rows if "strip" in r]
        result["pages"].append({
            "id": pid, "title": title, "image": f"{pid}.jpg", "image_size": web.size, "chars": total,
            "cer": cer, "extra": extra, "rows": rows, "scale": 0.45, "real": True,
            "note": f"A real scan: {cite}. Public domain, via Wikimedia Commons (National Library of Israel scan). "
                    f"Correct text: Hebrew Wikisource, validated, CC BY-SA 4.0.",
            "source": {k: src.get(k) for k in ("wikisource_page_url", "commons_file_url", "scan_licence", "transcription_licence")},
        })
        print(pid, len(refs), "lines,", len(segs), "found,", len(pairs), "matched;", cer)
    (a.out / "wikisource.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
