#!/usr/bin/env python3
"""Turn a ground-truth folder (NNNNNN.png + NNNNNN.gt.txt) into a ready-to-train
tesstrain-style layout, without tesstrain's Makefile.

Same recipes as tesstrain's Makefile for LANG_TYPE=RTL with START_MODEL set:
  - WordStr box per line, text in visual order WITHOUT bracket mirroring (fpdf2's
    BidiParagraph.get_reordered_string). tesstrain's generate_wordstr_box.py uses
    python-bidi get_display, which mirrors ( ) [ ] in RTL runs; Tesseract expects them
    unmirrored, and round 1 trained with mirrored boxes learned to output )85.0(.
  - lstmf per line: tesseract IMG BASE --psm 13 lstm.train
  - list.train / list.eval (90/10, seeded shuffle)
  - unicharset: start model's lstm-unicharset merged with the ground truth's
    (unicharset_extractor --norm_mode 3)
  - proto model: combine_lang_model --pass_through_recoder --lang_is_rtl with the
    stock wordlist / numbers / punc

Why not tesstrain's make: with 40k lines on a hard disk, make spent 11+ minutes resolving
pattern rules (five image-type alternatives per target) and built nothing.

Line direction: the box text is reordered with an explicit paragraph direction, RTL for
any line with Hebrew in it (as gen_lines.py renders it). Auto-detection would make a line
that starts with an English word LTR and the label would no longer match the image.

Brackets next to English (round 4): Tesseract treats a word with Latin letters and no
Hebrew as LTR and keeps its characters as recognised, so `(Check Point)` must be labelled
by glyph shape. In such words a bracket drawn mirrored (odd embedding level) is labelled
with its mirror image. Every other word keeps the unmirrored characters (rounds 2-3), which
is what fixed `)85.0(`. Checked with the stock script model: it outputs `(Ma Lookup ARCHIE
LOT)` correctly only when its raw output carries glyph shapes.

Layout written under --data-dir (DATA):
  DATA/start/MODEL.lstm, .lstm-unicharset, ...   the start model, unpacked
  DATA/MODEL/list.train, list.eval, unicharset    training lists and character set
  DATA/MODEL/MODEL.traineddata                    proto model for lstmtraining --traineddata
The .box and .lstmf files go next to the images in GT_DIR. --script-dir holds the
langdata_lstm script files: Hebrew.unicharset, Hebrew.xheights and radical-stroke.txt, plus
Latin.unicharset and Latin.xheights for heb_rivok (round 4). heb_rivok_lite (round 3) was
built without the Latin files; they change the character properties in the unicharset.

    python -I prep_train.py GT_DIR --model NAME --data-dir DATA --script-dir DIR \
        --start START.traineddata --words W --numbers N --puncs P [--jobs 6] [--limit N]
"""
import argparse
import os
import random
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fpdf.bidi import BidiParagraph
from fpdf.enums import TextDirection
from PIL import Image


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p


MIRROR = str.maketrans("()[]{}<>", ")(][}{><")


def visual_label(line: str) -> str:
    """Visual-order box text (see the module docstring for the bracket rule)."""
    rtl = any("\u0590" <= c <= "\u05ff" for c in line)
    chars = BidiParagraph(text=line, base_direction=TextDirection.RTL if rtl else TextDirection.LTR).reorder_resolved_levels()
    out, word = [], []
    def flush():
        txt = "".join(c.character for c in word)
        if any(c.isascii() and c.isalpha() for c in txt) and not any("\u05d0" <= c <= "\u05ea" for c in txt):
            txt = "".join(c.character.translate(MIRROR) if c.embedding_level % 2 else c.character for c in word)
        out.append(txt); word.clear()
    for c in chars:
        if c.character == " ":
            flush(); out.append(" ")
        else:
            word.append(c)
    flush()
    return "".join(out)


def box_and_lstmf(png: Path):
    base = png.with_suffix("")
    lstmf = base.with_suffix(".lstmf")
    if lstmf.exists() and b"IEND" in lstmf.read_bytes():      # complete (a killed run can leave halves)
        return str(lstmf)
    line = unicodedata.normalize("NFC", Path(f"{base}.gt.txt").read_text(encoding="utf-8").strip().split("\n")[0].strip())
    if not line:
        return None
    w, h = Image.open(png).size
    visual = visual_label(line)
    Path(f"{base}.box").write_text(f"WordStr 0 0 {w} {h} 0 #{visual}\n\t 0 0 {w} {h} 0\n", encoding="utf-8")
    p = subprocess.run(["tesseract", str(png), str(base), "--psm", "13", "lstm.train"],
                       capture_output=True, env=dict(os.environ, OMP_THREAD_LIMIT="1"))
    return str(lstmf) if lstmf.exists() and b"IEND" in lstmf.read_bytes() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gt")
    ap.add_argument("--model", required=True, help="name of the model being trained")
    ap.add_argument("--data-dir", required=True, help="working folder (lists, unicharset, proto model)")
    ap.add_argument("--script-dir", help="langdata_lstm script files (default DATA/langdata)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261106, help="shuffle seed for the train/eval split")
    ap.add_argument("--start", required=True, help="start traineddata (tessdata_best float model)")
    ap.add_argument("--words", required=True)
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--puncs", required=True)
    a = ap.parse_args()
    gt = Path(a.gt).resolve()
    DATA = Path(a.data_dir).resolve()
    script_dir = Path(a.script_dir) if a.script_dir else DATA / "langdata"
    out = DATA / a.model
    out.mkdir(parents=True, exist_ok=True)

    pngs = sorted(gt.glob("*.png"))
    if a.limit:
        pngs = pngs[:a.limit]
    print(f"{len(pngs)} lines: boxes + lstmf with {a.jobs} jobs", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        done = [x for x in ex.map(box_and_lstmf, pngs) if x]
    print(f"  {len(done)} lstmf ok, {len(pngs) - len(done)} failed", flush=True)
    random.Random(a.seed).shuffle(done)
    k = max(int(len(done) * 0.9), len(done) - 1000)     # eval list <= 1000 lines (it is re-run at every new best)
    (out / "list.train").write_text("\n".join(done[:k]) + "\n")
    (out / "list.eval").write_text("\n".join(done[k:]) + "\n")

    # start model parts: DATA/start/<model>.lstm, .lstm-unicharset, ...
    start_dir = DATA / "start"
    start_dir.mkdir(exist_ok=True)
    run(["combine_tessdata", "-u", a.start, str(start_dir / a.model)])
    all_gt = out / "all-gt.txt"
    with open(all_gt, "w", encoding="utf-8") as f:
        for png in pngs:
            f.write(Path(png.with_suffix("").as_posix() + ".gt.txt").read_text(encoding="utf-8").strip() + "\n")
    run(["unicharset_extractor", "--output_unicharset", str(out / "my.unicharset"), "--norm_mode", "3", str(all_gt)])
    run(["merge_unicharsets", str(start_dir / f"{a.model}.lstm-unicharset"), str(out / "my.unicharset"), str(out / "unicharset")])
    n_old = int(open(start_dir / f"{a.model}.lstm-unicharset", encoding="utf-8").readline())
    n_new = int(open(out / "unicharset", encoding="utf-8").readline())
    print(f"  unicharset {n_old} -> {n_new}", flush=True)
    run(["combine_lang_model", "--input_unicharset", str(out / "unicharset"), "--script_dir", str(script_dir),
         "--numbers", a.numbers, "--puncs", a.puncs,
         "--words", a.words, "--output_dir", str(DATA),
         "--pass_through_recoder", "--lang_is_rtl", "--lang", a.model])
    print(f"  proto model {DATA / a.model / (a.model + '.traineddata')}", flush=True)


if __name__ == "__main__":
    main()
