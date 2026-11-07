#!/usr/bin/env python3
"""The limit example: short mixed Hebrew/English lines, each read on its own as a single line.

Tesseract decides a single line's direction from its words, and when the English words are at
least as many as the Hebrew ones the line goes left-to-right, so the English comes out first.
paddleocr-hebrew assembles lines in Hebrew reading order. Lines written by us, rendered in IBM
Plex Sans Hebrew (not used in training). The page shows the first candidate, in this order, that
paddleocr-hebrew reads exactly and heb_rivok (psm 13) does not; every candidate's result is kept
in out/mixed.json.

    python -I mixed_example.py --fonts FONT_DIR --paddle-models DIR [--ocrmypdf-hebrew-src DIR] [--out out/]

FONT_DIR: as in make_examples.py (IBM Plex Sans Hebrew). paddleocr-hebrew runs through the
ocrmypdf-hebrew plugin: --paddle-models is its model folder (charset_v2f.txt, word-det/,
server-svtrv2/, ...), and the plugin is imported from --ocrmypdf-hebrew-src (its src/ folder)
or, without that flag, from the installed package. Needs opencv-python and onnxruntime.
"""
import argparse, json, subprocess, sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))      # python -I does not add the script's folder
from make_examples import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
TEXTS = [
    "איכות הרגולציה (The World Bank)",
    "הרצאה על Machine Learning",
    "דוח שנתי Annual Report 2025",
    "מחקר של Harvard Medical School",
    "תוכנת Adobe Acrobat Reader",
    "סדנת Python for Data Science",
    "שיתוף פעולה עם Tel Aviv University",
    "מדריך התקנה של Microsoft Office",
]


def render(text, font, path):
    d = ImageDraw.Draw(Image.new("L", (10, 10)))
    l, t, r, b = d.textbbox((0, 0), text, font=font, direction="rtl", language="he")
    img = Image.new("L", (r - l + 40, b - t + 28), 255)
    ImageDraw.Draw(img).text((20 - l, 14 - t), text, font=font, fill=0, direction="rtl", language="he")
    img.save(path)
    img.convert("RGB").save(path.with_suffix(".jpg"), quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", required=True, type=Path)
    ap.add_argument("--paddle-models", required=True, type=Path, help="paddleocr-hebrew model folder")
    ap.add_argument("--ocrmypdf-hebrew-src", type=Path, help="ocrmypdf-hebrew src/ folder (default: installed package)")
    ap.add_argument("--out", default=HERE / "out", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(a.fonts / "ibmplexsanshebrew_IBMPlexSansHebrew-Regular.ttf"), 31,
                              layout_engine=ImageFont.Layout.RAQM)
    if a.ocrmypdf_hebrew_src:
        sys.path.insert(0, str(a.ocrmypdf_hebrew_src))
    import cv2
    from ocrmypdf_hebrew import _reader
    from ocrmypdf_hebrew.layout import build_lines, lines_text
    reader = _reader(str(a.paddle_models), "word", 2)
    cands, chosen = [], None
    for i, text in enumerate(TEXTS, 1):
        png = a.out / f"mixed{i}.png"
        render(text, font, png)
        res = {"ref": text, "strip": png.with_suffix(".jpg").name, "out": {}}
        for psm in ("7", "13"):
            cmd = ["tesseract", "--tessdata-dir", str(HERE.parent / "tessdata"), str(png), "stdout",
                   "-l", "heb_rivok", "--psm", psm]
            res["out"][f"heb_rivok_psm{psm}"] = norm(subprocess.run(cmd, capture_output=True, text=True).stdout)
        out = reader.read_array(cv2.imread(str(png)), rec_mode="cascade")
        res["out"]["paddleocr_hebrew"] = norm(lines_text(build_lines(out["words"])))
        cands.append(res)
        ok = res["out"]["paddleocr_hebrew"] == norm(text) and res["out"]["heb_rivok_psm13"] != norm(text)
        print(f"{i} {'*' if ok else ' '} {text} | rivok13: {res['out']['heb_rivok_psm13']} | paddle: {res['out']['paddleocr_hebrew']}")
        if ok and chosen is None:
            chosen = res
    (a.out / "mixed.json").write_text(json.dumps(chosen | {"candidates": cands}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
