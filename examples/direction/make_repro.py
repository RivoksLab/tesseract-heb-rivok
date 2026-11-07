#!/usr/bin/env python3
"""Images that reproduce Tesseract's line-direction tie (ResultIterator::CurrentParagraphIsLtr):
a short Hebrew line whose English words are as many as its Hebrew ones comes out English first.

    python3 make_repro.py            # writes line_tie.png and page_heading.png next to this file
    tesseract line_tie.png stdout -l heb+eng --psm 7
    tesseract page_heading.png stdout -l heb+eng

Rendered in DejaVu Sans with Pillow + raqm (fonts-dejavu-core on Debian/Ubuntu).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def render(lines, name, size=32, width=1100):
    font = ImageFont.truetype(FONT, size, layout_engine=ImageFont.Layout.RAQM)
    img = Image.new("L", (width, 40 + int(size * 1.6) * len(lines)), 255)
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        _, _, r, _ = d.textbbox((0, 0), t, font=font, direction="rtl", language="he")
        d.text((width - 30 - r, 20 + i * int(size * 1.6)), t, font=font, fill=0, direction="rtl", language="he")
    img.save(HERE / name)


render(["הרצאה על Machine Learning"], "line_tie.png", width=560)
render(["דוח שנתי Annual Report", "", "החברה סיימה את השנה עם צמיחה של שמונה אחוזים בהכנסות,",
        "בזכות הביקוש הגובר למוצרים החדשים בשוק המקומי ובחו״ל."], "page_heading.png")
