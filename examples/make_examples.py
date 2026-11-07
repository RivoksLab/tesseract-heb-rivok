#!/usr/bin/env python3
"""Example pages for the heb_rivok web page and model card.

Seven short documents whose text we wrote for this purpose, set in fonts the model never saw
in training, damaged to look scanned, faxed, photocopied or photographed, then read as whole
pages with Tesseract's default page mode (psm 3) by each model. Every line is kept and shown:
nothing is picked after seeing the results.

    python -I make_examples.py --fonts FONT_DIR [--stock STOCK_DIR] [--out out/]

FONT_DIR holds (file names as used here):
  ibmplexsanshebrew_IBMPlexSansHebrew-Regular.ttf  IBM Plex Sans Hebrew, SIL OFL (google/fonts
                                                   ofl/ibmplexsanshebrew/IBMPlexSansHebrew-Regular.ttf)
  bonanova_BonaNova-Regular.ttf                    Bona Nova, SIL OFL (google/fonts ofl/bonanova/)
  MiriamCLM-Book.otf, SimpleCLM-Medium.ttf, FrankRuehlCLM-Medium.otf, NachlieliCLM-Light.otf,
  DavidCLM-Medium.otf                              Culmus, GPL-2 with font exception (Debian/Ubuntu
                                                   package fonts-culmus 0.140, /usr/share/fonts/truetype/culmus/)
Models: heb_rivok and heb_rivok_lite from ../tessdata; Ubuntu's heb and eng from Tesseract's own
tessdata folder (apt install tesseract-ocr-heb tesseract-ocr-eng; --system-tessdata to point
elsewhere). --stock adds Google's other Hebrew models (rows whose file is missing are skipped):
  STOCK_DIR/best/heb.traineddata           tessdata_best heb
  STOCK_DIR/script_best/Hebrew.traineddata tessdata_best script/Hebrew
  STOCK_DIR/script_fast/Hebrew.traineddata tessdata_fast script/Hebrew
Every model that is used with +eng gets the same eng.traineddata (Ubuntu's, = tessdata_fast eng).

Writes out/<page>.png (what the models read), out/<page>.jpg (web copy), out/<page>_L<n>.jpg
(line strips) and out/examples.json (reference, each model's text per line, CER).
Scoring folds typographic variants (״/", ׳/', ־/-, dashes) so stock models are not charged
for them; the page shows them in a lighter colour.
"""
import argparse, atexit, json, os, random, re, shutil, subprocess, sys, tempfile, unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
REL = HERE.parent


def add_model_args(ap):
    ap.add_argument("--stock", type=Path, help="folder with Google's other Hebrew models (see the module docstring)")
    ap.add_argument("--system-tessdata", type=Path, help="tessdata folder with Ubuntu's heb and eng "
                    "(default: the folder `tesseract --list-langs` reports)")


def system_tessdata():
    out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    m = re.search(r'"([^"]+)"', out.stdout + out.stderr)
    if not m:
        sys.exit("cannot find Tesseract's tessdata folder; pass --system-tessdata")
    return Path(m.group(1))


def build_models(a):
    """(key, label, tessdata dir or None for Tesseract's own, languages), in the published order."""
    sysdir = a.system_tessdata or system_tessdata()
    eng = sysdir / "eng.traineddata"
    tmp = Path(tempfile.mkdtemp(prefix="heb_rivok_examples_"))
    atexit.register(shutil.rmtree, tmp, True)

    def folder(name, *files):                       # one tessdata folder per combination, by symlinks
        d = tmp / name
        d.mkdir()
        for f in files:
            if not f.is_file():
                sys.exit(f"missing model file: {f}")
            (d / f.name).symlink_to(f.resolve())
        return d
    stock = a.stock
    has = lambda rel: stock is not None and (stock / rel).is_file()
    models = [("ubuntu_heb", "Tesseract heb (Ubuntu)", a.system_tessdata, "heb+eng")]
    if has("best/heb.traineddata"):
        models.append(("best_heb", "Google tessdata_best heb", folder("best", stock / "best/heb.traineddata", eng), "heb+eng"))
    models.append(("round3", "Round 3 (from tessdata_best heb)",
                   folder("lite", REL / "tessdata/heb_rivok_lite.traineddata", eng), "heb_rivok_lite+eng"))
    if has("script_best/Hebrew.traineddata"):
        models.append(("script_best", "Google script/Hebrew (best)", stock / "script_best", "Hebrew"))
    if has("script_fast/Hebrew.traineddata"):
        models.append(("script_fast", "Google script/Hebrew (fast)", stock / "script_fast", "Hebrew"))
    models.append(("heb_rivok", "heb_rivok", REL / "tessdata", "heb_rivok"))
    return models

PAGES = [
    {
        "id": "invoice",
        "title": "An invoice, scanned",
        "font": "ibmplexsanshebrew_IBMPlexSansHebrew-Regular.ttf",
        "font_name": "IBM Plex Sans Hebrew",
        "damage": "scan",
        "lines": [
            "חשבונית מס/קבלה מס׳ 20417",
            "תאריך: 03.11.2026",
            "לכבוד: גב׳ מיכל אברהם, רח׳ הרצל 12, חיפה",
            "תיקון מזגן בחדר השינה והחלפת מסנן: 450.00 ₪",
            "ביקור טכנאי (כולל נסיעה): 180.00 ₪",
            "סה״כ לפני מע״מ: 630.00 ₪",
            "מע״מ 18%: 113.40 ₪",
            "סה״כ לתשלום: 743.40 ₪",
            "התשלום יבוצע בהעברה בנקאית לחשבון 12-345-678901 עד סוף החודש.",
            "לשאלות: 04-8123456 או בדוא״ל service@example.co.il",
            "תודה שבחרתם בנו!",
        ],
    },
    {
        "id": "notice",
        "title": "A building notice, photocopied",
        "font": "bonanova_BonaNova-Regular.ttf",
        "font_name": "Bona Nova",
        "damage": "photocopy",
        "lines": [
            "הודעה לדיירי הבניין",
            "ביום ג׳, ה־10 בנובמבר, יתבצעו עבודות שיפוץ בחדר המדרגות.",
            "העבודות יחלו בשעה 08:00 ויסתיימו לא יאוחר מ־16:00.",
            "במהלך העבודות המעלית לא תפעל, ולכן נבקש מכם לתכנן את יומכם בהתאם.",
            "ועד הבית מתנצל על אי־הנוחות ומודה לכם על שיתוף הפעולה.",
            "נציג הקבלן, מר דוד שמעוני, זמין בטלפון 052-7654321.",
            "העבודות מבוצעות לפי תקן ISO 9001 ובאישור מהנדס הבניין.",
            "בברכה,",
            "ועד הבית, רחוב הגפן 7, תל־אביב־יפו",
        ],
    },
    # Five more documents, written 2026-11-07 around the error types stock Tesseract makes most
    # on real document lines (digits, full stops, ם/ס, maqaf/geresh). Every name, number and
    # organisation here is made up.
    {
        "id": "bank",
        "title": "A bank letter, faxed",
        "font": "MiriamCLM-Book.otf",
        "font_name": "Miriam CLM",
        "damage": "fax",
        "lines": [
            "בנק הדוגמה בע״מ, סניף 642, רח׳ יפו 97, ירושלים",
            "לכבוד: מר יוסף כהן",
            "הנדון: עדכון פרטי חשבון מס׳ 99-642-0458321",
            "יתרת החשבון ליום 31.10.2026 עומדת על 18,452.37 ₪.",
            "הלוואה מס׳ 7781-22 תיפרע ב־36 תשלומים של 1,240.00 ₪.",
            "הריבית השנתית: פריים + 1.5% (סה״כ 7.5%).",
            "לבירורים: 03-1234567, שלוחה 4, בימים א׳-ה׳ בין 08:30 ל־15:00.",
            "אנא שמרו מכתב זה לתיעוד.",
            "בכבוד רב,",
            "מחלקת שירות לקוחות",
        ],
    },
    {
        "id": "medical",
        "title": "A medical visit summary, scanned",
        "font": "SimpleCLM-Medium.ttf",
        "font_name": "Simple CLM",
        "damage": "scan",
        "lines": [
            "מרפאת הדוגמה, מחלקה פנימית ב׳",
            "סיכום ביקור מיום 12.08.2026",
            "שם המטופלת: רחל לוי, גיל 67",
            "אבחנה: יתר לחץ דם, סוכרת סוג 2.",
            "טיפול: אמלודיפין 5 מ״ג פעם ביום, מטפורמין 850 מ״ג פעמיים ביום.",
            "לחץ דם: 145/90, דופק: 78, חום: 36.8.",
            "המלצות: בדיקת HbA1c בעוד 3 חודשים ומעקב אצל רופא המשפחה.",
            "ביקורת חוזרת במרפאה ביום 15.11.2026 בשעה 10:20.",
            "ד״ר אבי שגיא, מ״ר 34521",
        ],
    },
    {
        "id": "rental",
        "title": "A rental agreement, photocopied",
        "font": "FrankRuehlCLM-Medium.otf",
        "font_name": "Frank Ruehl CLM",
        "damage": "photocopy",
        "lines": [
            "הסכם שכירות בלתי מוגנת",
            "שנערך ונחתם ביום 15.07.2026",
            "בין: דנה אורן (להלן: \"המשכירה\")",
            "לבין: עמית גל (להלן: \"השוכר\")",
            "1. המשכירה משכירה לשוכר דירה בת 3 חדרים ברח׳ האלה 14, דירה 6, באר שבע.",
            "2. תקופת השכירות: 12 חודשים, מיום 01.08.2026 ועד ליום 31.07.2027.",
            "3. דמי השכירות: 4,350 ₪ לחודש, שישולמו עד ה־10 בכל חודש.",
            "4. השוכר ימסור למשכירה ערבות בנקאית בסך 13,050 ₪.",
            "5. הצדדים מצהירים כי קראו הסכם זה והבינו את תוכנו.",
        ],
    },
    {
        "id": "water",
        "title": "A water bill, photographed",
        "font": "NachlieliCLM-Light.otf",
        "font_name": "Nachlieli CLM",
        "damage": "phone",
        "lines": [
            "תאגיד המים העירוני: חשבון תקופתי",
            "מספר משלם: 4402-118-9",
            "תקופת החשבון: 01.09.2026 עד 31.10.2026",
            "קריאה קודמת: 1,284 מ״ק. קריאה נוכחית: 1,309 מ״ק.",
            "צריכה: 25 מ״ק לפי 7.92 ₪ למ״ק, סך הכול 198.00 ₪",
            "דמי שירות קבועים: 34.50 ₪",
            "סה״כ לתשלום כולל מע״מ: 274.17 ₪",
            "מועד אחרון לתשלום: 20.11.2026",
            "לתשלום באתר: www.mayim-example.co.il",
        ],
    },
    {
        "id": "news",
        "title": "A local news item, scanned",
        "font": "DavidCLM-Medium.otf",
        "font_name": "David CLM",
        "damage": "scan",
        "lines": [
            "מועצת העיר אישרה אמש את תקציב העירייה לשנת 2027.",
            "בעד הצביעו 17 חברי מועצה, ונגד 6.",
            "היקף התקציב עומד על 1.24 מיליארד ₪, גידול של 4.2% לעומת השנה הקודמת.",
            "ראש העיר אמר כי \"זהו תקציב של צמיחה ואחריות\".",
            "באופוזיציה טענו כי התקציב אינו מקצה די משאבים לחינוך ולתחבורה.",
            "לפי הנתונים, 38% מהתקציב יופנו לחינוך ו־12% לשיפור התחבורה הציבורית.",
            "התקציב ייכנס לתוקף ב־1 בינואר.",
        ],
    },
]

W, MARGIN, SIZE, PITCH, SS = 1654, 150, 31, 50, 2   # A4 width at 200 dpi, ~11 pt text


def render(page, font_dir):
    """Render at 2x, right-aligned; return the 200 dpi image and each line's band."""
    font = ImageFont.truetype(str(font_dir / page["font"]), SIZE * SS, layout_engine=ImageFont.Layout.RAQM)
    big = ImageFont.truetype(str(font_dir / page["font"]), int(SIZE * 1.35) * SS, layout_engine=ImageFont.Layout.RAQM)
    n = len(page["lines"])
    H = 2 * MARGIN // 2 + int(PITCH * 1.3) + PITCH * n + MARGIN // 2
    img = Image.new("L", (W * SS, H * SS), 255)
    d = ImageDraw.Draw(img)
    y, bands = MARGIN // 2, []
    for i, text in enumerate(page["lines"]):
        f = big if i == 0 else font
        l, t, r, b = d.textbbox((0, 0), text, font=f, direction="rtl", language="he")
        x = W * SS - MARGIN * SS - r
        d.text((x, y * SS), text, font=f, fill=0, direction="rtl", language="he")
        bands.append((y + t // SS, y + b // SS, (x + l) // SS, (x + r) // SS))
        y += int(PITCH * 1.3) if i == 0 else PITCH
    return img.resize((W, H), Image.LANCZOS), bands


def damage(img, kind, rng):
    a = np.asarray(img).astype(np.float32)
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    if kind == "scan":
        img = Image.fromarray(a.astype(np.uint8)).rotate(0.35, resample=Image.BICUBIC, fillcolor=255)
        a = np.asarray(img).astype(np.float32)
        tone = Image.fromarray((nprng.random((8, 12)) * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)
        a = a * (0.90 + 0.06 * np.asarray(tone) / 255) + nprng.normal(0, 6, a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
        quality = 45
    elif kind == "fax":  # half vertical resolution (fax "standard" mode), hard black and white
        img = Image.fromarray(a.astype(np.uint8)).rotate(0.25, resample=Image.BICUBIC, fillcolor=255)
        w, h = img.size
        a = np.asarray(img.resize((w, h // 2), Image.BILINEAR)).astype(np.float32) + nprng.normal(0, 12, (h // 2, w))
        img = Image.fromarray(np.where(a < 150, 0, 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
        return img
    elif kind == "phone":  # soft focus, uneven light, sensor noise, strong JPEG
        img = Image.fromarray(a.astype(np.uint8)).rotate(0.8, resample=Image.BICUBIC, fillcolor=255)
        w, h = img.size
        img = img.filter(ImageFilter.GaussianBlur(1.1))
        yy, xx = np.mgrid[0:h, 0:w]
        light = 0.72 + 0.28 * (1 - ((xx / w - 0.3) ** 2 + (yy / h - 0.2) ** 2) ** 0.5)
        a = np.asarray(img).astype(np.float32) * light + nprng.normal(0, 7, (h, w))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        quality = 50
    else:  # photocopy: lose detail, then threshold with noise (broken and swollen strokes), speckle
        img = Image.fromarray(a.astype(np.uint8)).rotate(-0.5, resample=Image.BICUBIC, fillcolor=255)
        w, h = img.size
        img = img.resize((int(w * 0.62), int(h * 0.62)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(0.9))
        a = np.asarray(img).astype(np.float32) + nprng.normal(0, 22, (h, w))
        a = np.where(a < 168, 25.0, 238.0)
        spk = nprng.random((h, w))
        a[spk < 0.0007] = 30
        img = Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))
        quality = 40
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def tesseract(png, model):
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
        top, height = int(f[7]), int(f[9])
        lines.setdefault(k, {"words": [], "ys": []})
        lines[k]["words"].append(f[11])
        lines[k]["ys"].append(top + height / 2)
    return [(" ".join(v["words"]), sum(v["ys"]) / len(v["ys"])) for v in lines.values()]


FOLD = str.maketrans({"״": '"', "׳": "'", "־": "-", "–": "-", "—": "-", "“": '"', "”": '"', "„": '"', "’": "'", "‘": "'"})


BIDI_MARKS = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩"))


def norm(s):
    """NFC, no invisible direction marks (Tesseract adds U+200F after LTR words), single spaces,
    no space before punctuation (old print sets one before commas; modern transcriptions don't,
    so a model that copies the print faithfully would be charged for it)."""
    s = " ".join(unicodedata.normalize("NFC", s).translate(BIDI_MARKS).split())
    return re.sub(r" +([,.;:!?])", r"\1", s)


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", required=True, type=Path, help="folder holding the fonts listed above")
    ap.add_argument("--out", default=HERE / "out", type=Path)
    add_model_args(ap)
    a = ap.parse_args()
    MODELS = build_models(a)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261107)
    result = {"models": [{"key": m[0], "label": m[1]} for m in MODELS], "pages": []}
    for page in PAGES:
        clean, bands = render(page, a.fonts)
        img = damage(clean, page["damage"], rng)
        png = a.out / f"{page['id']}.png"
        img.save(png)
        img.convert("RGB").save(a.out / f"{page['id']}.jpg", quality=82)
        with ThreadPoolExecutor(3) as ex:
            outs = dict(zip([m[0] for m in MODELS], ex.map(lambda m: tesseract(png, m), MODELS)))
        centres = [(bd[0] + bd[1]) / 2 for bd in bands]
        rows = []
        for i, (ref, (t, b, x0, x1)) in enumerate(zip(page["lines"], bands)):
            pad = 16
            strip = img.crop((max(0, x0 - pad - 10), max(0, t - pad), min(W, x1 + pad), min(img.height, b + pad)))
            strip.convert("RGB").save(a.out / f"{page['id']}_L{i + 1}.jpg", quality=85)
            rows.append({"ref": ref, "strip": f"{page['id']}_L{i + 1}.jpg", "size": strip.size, "out": {}, "edits": {}})
        for key, lines in outs.items():
            got = [[] for _ in bands]
            for text, y in lines:     # each Tesseract line goes to the nearest reference line
                got[min(range(len(centres)), key=lambda j: abs(centres[j] - y))].append(text)
            for row, parts in zip(rows, got):
                hyp = norm(" ".join(parts))
                row["out"][key] = hyp
                row["edits"][key] = lev(norm(row["ref"]).translate(FOLD), hyp.translate(FOLD))
        total = sum(len(norm(r["ref"])) for r in rows)
        cer = {k: round(100 * sum(r["edits"][k] for r in rows) / total, 2) for k in outs}
        result["pages"].append({k: page[k] for k in ("id", "title", "font_name", "damage")} | {
            "image": f"{page['id']}.jpg", "image_size": img.size, "chars": total, "cer": cer, "rows": rows})
        print(page["id"], cer)
    (a.out / "examples.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
