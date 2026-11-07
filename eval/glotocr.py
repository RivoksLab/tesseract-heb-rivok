#!/usr/bin/env python3
"""GlotOCR-bench, Hebrew split: mean CER of a Tesseract model on both image variants.

GlotOCR-bench (Hugging Face cis-lmu/GlotOCR-bench, file v1.0/Hebr_000.parquet, 100 lines in
random Google Fonts, many vowelled) comes with an evaluation-only licence: it is downloaded at
run time, never stored in this repository and never used for training. The dataset is gated:
accept its terms on Hugging Face and log in (`hf auth login`, or set HF_TOKEN) before the
first run.

Each line image is read with `tesseract --psm 6` (one Tesseract process per image, one
thread) and scored with the benchmark's own metric (metrics.glot_cer); "no marks" also strips
niqqud from both sides.

    python -I glotocr.py --tessdata ../tessdata --lang heb_rivok
    python -I glotocr.py --lang heb                       # the system's heb (Ubuntu: tessdata_fast)
    python -I glotocr.py --tessdata DIR --lang L --data-dir DIR --jobs 2 --out preds.jsonl

--data-dir: a folder holding Hebr_000.parquet (or v1.0/Hebr_000.parquet); if the file is not
there it is downloaded into it. Without --data-dir the Hugging Face cache is used.
Needs: tesseract, Python with pyarrow, huggingface_hub, Pillow, python-bidi.
"""
import argparse
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402

REPO = "cis-lmu/GlotOCR-bench"
FILE = "v1.0/Hebr_000.parquet"
REVISION = "6a4b2d6785399357b46af6c989173ba3d9e9be8e"     # the dataset revision behind the published numbers
SHA256 = "e5fa3ef80f2d7d0f551a66df3e4805915b701e06b1c5ee24f430e8e47fca0727"
VARIANTS = ("img_plain", "img_old_document")


def parquet_path(data_dir, revision):
    if data_dir:
        d = Path(data_dir)
        for p in (d / "Hebr_000.parquet", d / FILE):
            if p.is_file():
                return p
    from huggingface_hub import hf_hub_download
    kw = {"local_dir": data_dir} if data_dir else {}
    return Path(hf_hub_download(REPO, FILE, repo_type="dataset", revision=revision, **kw))


def tesseract(png_bytes, tessdata, lang, psm):
    cmd = ["tesseract", "stdin", "stdout", "-l", lang, "--psm", str(psm)]
    if tessdata:
        cmd[1:1] = ["--tessdata-dir", str(tessdata)]
    p = subprocess.run(cmd, input=png_bytes, capture_output=True, env=dict(os.environ, OMP_THREAD_LIMIT="1"))
    err = p.stderr.decode("utf-8", "replace")
    if p.returncode and ("Error opening data file" in err or "Failed loading language" in err):
        sys.exit(f"tesseract cannot load the model: {' '.join(cmd)}\n{err[-1000:]}")
    return " ".join(p.stdout.decode("utf-8", "replace").split())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tessdata", help="folder holding LANG.traineddata (default: Tesseract's own)")
    ap.add_argument("--lang", required=True, help="model name(s) for -l, e.g. heb_rivok or heb")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=2, help="images read in parallel (one thread each)")
    ap.add_argument("--data-dir", help="local folder for the parquet file (see above)")
    ap.add_argument("--revision", default=REVISION, help="dataset revision to download")
    ap.add_argument("--out", help="write every prediction to this JSONL file")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image
    path = parquet_path(a.data_dir, a.revision)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != SHA256:
        print(f"warning: {path} is not the file behind the published numbers (sha256 {sha})", file=sys.stderr)
    rows = pq.read_table(path).to_pylist()

    results, records = {}, []
    for col in VARIANTS:
        def one(r):
            png = io.BytesIO()
            Image.open(io.BytesIO(r[col]["bytes"])).convert("RGB").save(png, "PNG")
            gt, pred = M.normalize_text(r["text"]), tesseract(png.getvalue(), a.tessdata, a.lang, a.psm)
            return {"variant": col, "id": r["id"], "gt": gt, "pred": pred, "cer": M.glot_cer(pred, gt),
                    "cer_nomarks": M.glot_cer(M.strip_marks(pred), M.strip_marks(gt))}
        with ThreadPoolExecutor(a.jobs) as ex:
            res = list(ex.map(one, rows))
        records += res
        results[col] = (100 * statistics.mean(x["cer"] for x in res), 100 * statistics.mean(x["cer_nomarks"] for x in res), len(res))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"GlotOCR-bench Hebrew, {a.lang} (psm {a.psm}), mean CER:")
    for col, (cer, nomarks, n) in results.items():
        print(f"  {col:17s} n={n}  {cer:5.1f}%   (no marks {nomarks:5.1f}%)")


if __name__ == "__main__":
    main()
