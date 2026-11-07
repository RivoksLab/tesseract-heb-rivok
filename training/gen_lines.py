#!/usr/bin/env python3
"""Synthetic Hebrew line images + ground truth for fine-tuning Tesseract.

Default profile (heb_rivok_lite, round 3): Hebrew + digits + symbols, no Latin letters.
Text mix per line:
  ~50%  a slice of Tesseract's own heb.training_text (keeps what the model knows,
        including its niqqud lines)
  ~32%  document-style lines: wordlist words + numbers, prices, dates, phones,
        gershayim/geresh abbreviations, prefixed numbers (ב-2024), maqaf, en dash
  ~18%  short crops: one or two tokens (real word/line crops are often this short)

--mixed (heb_rivok, round 4; start model script/Hebrew, which reads Latin):
  ~38% training-text slices, ~22% Hebrew document lines, ~20% Hebrew lines with English
  inside (terms in parentheses, ל-Facebook, names, acronyms, codes, URLs, citations;
  some lines start with English), ~7% English-only lines (so English is not forgotten),
  ~13% short crops (some English). Keeps the 12 niqqud marks script/Hebrew knows.
--clean: no degradation and no neighbour fragments (for checking box conventions).
--legacy-r3: the generator exactly as it was for round 3 (heb_rivok_lite). Round 4 added two
  fixes: ink thinning is skipped when it would erase a light font, and a line that comes out
  nearly blank is degraded again (round 3 has ~3% blank images with text labels); also, an
  empty neighbour line no longer costs a random draw. --legacy-r3 undoes all three. Fonts that
  draw lowercase Latin as capitals are not used for lowercase Latin (no effect without Latin).

Rendering: Pillow + raqm (logical text in, correct BiDi and niqqud placement out), each line
in a font whose cmap covers every character. Degradation imitates scans: paper tone, blur,
down/up-sampling, noise, JPEG, ink thinning/thickening, slight rotation, speckles, old
letterpress, underlines, fragments of neighbouring lines.

Inputs:
  --langdata DIR   heb.wordlist, heb.training_text (and eng.wordlist, eng.training_text for
                   --mixed) from tesseract-ocr/langdata_lstm (fetch_inputs.sh)
  --fonts FILE     font list (fonts.txt: first column = file name, in this order)
  --font-dir DIR   where those files are (repeatable; fetch_fonts.sh puts them in one folder)
Each line is seeded on its own (seed * 1000003 + index), so the output does not depend on
--jobs, and --start/--n regenerate any slice of a run.

Output: OUT/NNNNNN.png + OUT/NNNNNN.gt.txt (logical order, NFC), tesstrain layout.

    python -I gen_lines.py OUT --langdata DIR --fonts fonts.txt --font-dir DIR \\
        --n 60000 --seed 4 --mixed [--jobs 6]
"""
import argparse
import io
import random
import re
import sys
import unicodedata
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

# display / decorative faces (matched on the file name): kept for variety but 1/4 as likely
DISPLAY = re.compile(r"Amatic|Karantina|Bellefair|Secular|Suez|Rubik.*(Mono|Glitch|Wet|Puddles|Bubbles)|Rashi|Unifont", re.I)
BIDI_CTRL = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩"))
LATIN = re.compile(r"[A-Za-z]")
# Target character set: the stock model's 66 real characters + the additions in scope.
BASE = ("0123456789,.?-\"':;()/%+!*\\|>][=<₪"
        "אבגדהוזחטיכךלמםנןסעפףצץקרשת" "\u05b0\u05b7\u05b8\u05bc\u05b4\u05b6")
ADDED = "\u05be\u05f3\u05f4\u2013\u2022€&@_"          # maqaf geresh gershayim en-dash bullet € & @ _
ALLOWED = set(BASE + ADDED + " ")
MARKS = re.compile(r"[\u0591-\u05c7]")
UNSUPPORTED_MARKS = re.compile("[" + "".join(chr(c) for c in range(0x0591, 0x05C8) if chr(c) not in ALLOWED) + "]")
# --mixed: script/Hebrew's extra marks (holam, tsere, shin/sin dot, hataf patah, rafe),
# Latin letters, and the symbols it knows that occur in mixed documents
MIXED_EXTRA = ("\u05b9\u05b5\u05c1\u05c2\u05b2\u05bf" + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
               + "$#~{}\u201c\u201d\u2019\u2014\u00b0\u00a7\u00a9\u00ae\u2122\u00e9")
MIXED = False


def set_profile(mixed):
    """Switch the module to the round-4 character set (called in every worker)."""
    global MIXED, ALLOWED, UNSUPPORTED_MARKS
    MIXED = mixed
    if mixed:
        ALLOWED = set(BASE + ADDED + MIXED_EXTRA + " ")
        UNSUPPORTED_MARKS = re.compile("[" + "".join(chr(c) for c in range(0x0591, 0x05C8) if chr(c) not in ALLOWED) + "]")

GERSHAYIM = ['ש"ח', 'עו"ד', 'בע"מ', 'מע"מ', 'סה"כ', 'ת"א', 'ד"ר', 'רו"ח', 'צה"ל', 'ארה"ב', 'חו"ל',
             'בס"ד', 'נ"ל', 'מנכ"ל', 'יו"ר', 'ע"י', 'ע"פ', 'כנ"ל', 'בי"ס', 'ת"ז', 'ח"פ', 'ע"ר', 'תשפ"ה']
GERESH = ["מס'", "עמ'", "רח'", "יח'", "גב'", "פרופ'", "טל'", "שכ'", "ד'", "ג'", "ה'", "סנט'", "מח'"]
LABELS = ["טלפון", "פקס", "כתובת", "תאריך", "סכום", "מספר", "הערות", "שם", "תעודת זהות", "חשבון",
          "סניף", "בנק", "מחיר", "כמות", "תיאור", "יחידה", "הנחה", "לתשלום", "סה\"כ לתשלום", "נושא"]
PREFIX = ["ב", "ל", "מ", "ה", "ו", "כ", "ש", "וב", "ול", "מה", "שב"]


def load_words(path, n=60000):
    out = []
    for w in open(path, encoding="utf-8"):
        w = w.strip()
        if w and not LATIN.search(w):
            out.append(w)
        if len(out) >= n:
            break
    return out


def load_training_text(path):
    """Tesseract's own training text. Vowelled lines carry marks outside the model's
    character set (holam, tsere, shin dot, cantillation, sof pasuq); round 2 dropped
    those lines, which removed most niqqud and cost niqqud output on GlotOCR. Now only
    the unsupported marks are stripped before rendering, so image and label agree and
    the six supported marks keep being trained."""
    return [UNSUPPORTED_MARKS.sub("", l.translate(BIDI_CTRL)).strip() for l in open(path, encoding="utf-8")
            if l.strip() and not LATIN.search(l)]


# ----------------------------------------------------------------- English (--mixed)
ACRONYMS = ["CEO", "CTO", "CFO", "PDF", "USB", "GPS", "OECD", "IMF", "WHO", "UN", "EU", "USA", "UK", "NATO", "NASA",
            "AI", "IT", "HR", "VAT", "GDP", "CPI", "ISO", "IEEE", "API", "URL", "HTML", "SQL", "PEG", "RIA", "MRI", "CT",
            "DNA", "RNA", "ICU", "PhD", "MBA", "B.A.", "M.A.", "B.Sc.", "Ltd.", "Inc.", "LLC", "NIS", "USD", "EUR", "KPI",
            "ROI", "SaaS", "IPO", "ETF", "S&P 500", "NASDAQ", "Wi-Fi", "4G", "5G", "3D", "TV", "GB", "MB", "km", "kg",
            "mg", "ml", "cm", "mm", "pH", "COVID-19", "BMI", "ADHD", "OK", "FAQ", "R&D", "B2B", "CRM", "ERP", "IoT"]
BRANDS = ["Google", "Microsoft", "Apple", "Facebook", "WhatsApp", "Instagram", "ChatGPT", "OpenAI", "Amazon",
          "Netflix", "Zoom", "Excel", "Word", "PowerPoint", "Outlook", "Windows", "Android", "iPhone", "iPad",
          "LinkedIn", "YouTube", "Twitter", "Intel", "Nvidia", "Tesla", "Waze", "Mobileye", "Check Point", "Teva",
          "Wix", "monday.com", "Fiverr", "Freedom House", "The World Bank", "Bank of Israel", "Harvard", "MIT",
          "Oxford", "Cambridge", "Priority", "SAP", "Oracle", "Salesforce", "GitHub", "Python", "Java", "Linux",
          "Need to Know", "Start-Up Nation", "Times New Roman", "Arial", "Gmail", "Telegram", "TikTok", "Spotify"]
SITES = [".co.il", ".com", ".org.il", ".gov.il", ".ac.il", ".org", ".net"]


def eng_word(eng):
    w = random.choice(eng[:30000]) if random.random() < 0.8 else random.choice(eng)
    r = random.random()
    return w.capitalize() if r < 0.3 else w.upper() if r < 0.38 else w


def eng_term(eng):
    r = random.random()
    if r < 0.22:
        return random.choice(ACRONYMS)
    if r < 0.42:
        return random.choice(BRANDS)
    if r < 0.62:
        return " ".join(eng_word(eng) for _ in range(random.randint(1, 4)))
    if r < 0.72:                                    # codes, models, references
        L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return random.choice([
            "".join(random.choices(L, k=random.randint(2, 3))) + r_int(100000, 999999999),
            "".join(random.choices(L, k=3)) + "-" + r_int(1000, 999999),
            random.choice(["ISO", "IEC", "EN", "IS"]) + " " + r_int(100, 99999),
            random.choice(L) + r_int(1, 99), random.choice(L) + "-" + r_int(100, 999)])
    if r < 0.82:                                    # web
        w = random.choice(eng[:5000]).lower()
        return random.choice(["www." + w + random.choice(SITES), w + "@" + random.choice(eng[:3000]).lower() + random.choice(SITES),
                              "https://" + w + random.choice(SITES) + "/" + random.choice(eng[:3000]).lower()])
    if r < 0.92:                                    # personal names, initials
        return f"{eng_word(eng).capitalize()} {random.choice('ABCDEFGHJKLMNPRSTW')}. {eng_word(eng).capitalize()}"
    return f"{random.randint(1, 999)} {random.choice(['GB', 'MB', 'km', 'kg', 'mg', 'ml', 'cm', 'mm', 'USD', 'EUR', 'kW', 'Hz'])}"


def citation(eng):
    name = f"{eng_word(eng).capitalize()} {random.choice('ABCDEFGHJKLMNPRSTW')}. {eng_word(eng).capitalize()}"
    title = " ".join(eng_word(eng).capitalize() for _ in range(random.randint(2, 6)))
    return random.choice([f"{name}, {title}, {random.randint(1950, 2026)}, pp. {random.randint(1, 300)}-{random.randint(301, 600)}",
                          f"{name} ({random.randint(1950, 2026)}). {title}.", f"{name}, {title} [eds.]",
                          f"{title}, Vol. {random.randint(1, 40)}, No. {random.randint(1, 12)}"])


def mixed_line(words, ttext, eng):
    if random.random() < 0.1:                       # bibliography-style line
        heb = " ".join(heb_word(words) for _ in range(random.randint(1, 4)))
        return random.choice([f"{heb}, {citation(eng)}", f"{citation(eng)}; {heb} {random.choice(GERESH)} {r_int(1, 400)}"])
    if random.random() < 0.5:
        toks = random.choice(ttext).split()
        k = random.randint(3, 9)
        i = random.randint(0, max(0, len(toks) - k))
        toks = [MARKS.sub("", t) or t for t in toks[i:i + k]]
    else:
        toks = doc_line(words).split()
    for _ in range(1 if random.random() < 0.7 else 2):
        t = eng_term(eng)
        r = random.random()
        t = (f"({t})" if r < 0.25 else random.choice(PREFIX) + ("־" if random.random() < 0.1 else "-") + t if r < 0.4
             else f'"{t}"' if r < 0.48 else t + ":" if r < 0.53 else t)
        pos = 0 if random.random() < 0.15 else random.randint(0, len(toks))
        toks.insert(pos, t)
    return " ".join(toks)


def english_line(eng, etext):
    if random.random() < 0.5:
        toks = random.choice(etext).split()
        k = random.randint(2, 8)
        i = random.randint(0, max(0, len(toks) - k))
        return " ".join(toks[i:i + k])
    lab = random.choice(["Invoice No.", "Tel:", "Fax:", "Total:", "Date:", "Ref:", "Email:", "Order", "Page", "Item"])
    return f"{lab} " + random.choice([number_token().replace("₪", "$"), eng_term(eng), date(), phone(),
                                      " ".join(eng_word(eng) for _ in range(random.randint(1, 5)))])


# ----------------------------------------------------------------- text makers
def r_int(a, b):
    return str(random.randint(a, b))


def money():
    v = random.choice([random.randint(1, 99), random.randint(100, 9999), random.randint(10000, 999999)])
    s = f"{v:,}" if random.random() < 0.7 else str(v)
    if random.random() < 0.6:
        s += "." + r_int(0, 99).zfill(2)
    return random.choice([s + " ₪", "₪" + s, "₪ " + s, s, s + ' ש"ח', "€" + s if random.random() < 0.1 else s])


def date():
    d, m, y = random.randint(1, 31), random.randint(1, 12), random.choice([random.randint(1948, 2030), random.randint(0, 99)])
    sep = random.choice(["/", ".", "-", "/"])
    ys = str(y).zfill(2)
    return f"{d:02d}{sep}{m:02d}{sep}{ys}" if random.random() < 0.6 else f"{d}{sep}{m}{sep}{ys}"


def phone():
    return random.choice([f"0{random.choice('234789')}-{r_int(1000000, 9999999)}",
                          f"05{r_int(0, 9)}-{r_int(1000000, 9999999)}",
                          f"05{r_int(0, 9)}-{r_int(100, 999)}-{r_int(1000, 9999)}",
                          f"*{r_int(1000, 9999)}", f"1-800-{r_int(100, 999)}-{r_int(100, 999)}"])


def number_token():
    return random.choice([
        r_int(1, 999), r_int(1000, 99999), f"{random.randint(1, 99)}%", f"{random.randint(0, 99)}.{random.randint(0, 9)}%",
        f"{random.randint(1, 20)}.{random.randint(1, 20)}", f"{random.randint(1, 9)}.{random.randint(1, 9)}.{random.randint(1, 9)}",
        r_int(100000000, 599999999), money(), date(), phone(), f"({random.randint(1, 30)})",
        f"{random.randint(1990, 2030)}{'–' if random.random() < 0.2 else '-'}{random.randint(1990, 2030)}", f"{random.randint(1, 50)}-{random.randint(51, 200)}",
    ])


def heb_word(words):
    w = random.choice(words[:20000]) if random.random() < 0.7 else random.choice(words)
    return MARKS.sub("", w) or w                    # documents are unvocalised


def doc_token(words):
    r = random.random()
    if r < 0.50:
        return heb_word(words)
    if r < 0.66:
        return number_token()
    if r < 0.74:
        return random.choice(GERSHAYIM)
    if r < 0.80:
        return random.choice(GERESH)
    if r < 0.88:                                    # prefixed number: ב-2024, ל-20%
        return random.choice(PREFIX) + random.choice(["-"] * 8 + ["־", "", ""]) + random.choice(
            [r_int(1, 2030), f"{random.randint(1, 99)}%", f"{random.randint(1, 9)}.{random.randint(0, 9)}", date()])
    if r < 0.90:                                    # compound: hyphen mostly, real maqaf rarely
        return heb_word(words) + ("־" if random.random() < 0.2 else "-") + heb_word(words)
    return random.choice(["(" + heb_word(words) + ")", '"' + heb_word(words) + '"', "״" + heb_word(words) + "״",
                          "-", "•", "&", "_____", "...", heb_word(words) + "?", heb_word(words) + "!",
                          heb_word(words) + ";", "[" + r_int(1, 99) + "]", "/", "+" + r_int(1, 99), "@"])


def doc_line(words):
    kind = random.random()
    if kind < 0.15:                                 # label: value
        lab = random.choice(LABELS) + random.choice([":", ": ", " :"])
        val = random.choice([number_token(), phone(), date(), money(), " ".join(heb_word(words) for _ in range(random.randint(1, 3)))])
        return f"{lab} {val}".replace(":  ", ": ")
    if kind < 0.22:                                 # list item
        head = random.choice(["•", "-", f"{random.randint(1, 20)}.", f"({random.choice('אבגדהוזחט')})",
                              f"{random.choice('אבגדהוזחט')}.", f"{random.randint(1, 9)}.{random.randint(1, 9)}"])
        return head + " " + " ".join(doc_token(words) for _ in range(random.randint(2, 8)))
    n = random.randint(2, 10)
    toks = [doc_token(words) for _ in range(n)]
    s = ""
    for i, t in enumerate(toks):
        s += ("" if i == 0 else " ") + t
        if i < n - 1 and random.random() < 0.12 and not t[-1:] in ".,:;":
            s += random.choice([",", ",", ".", ":", ";"])
    if random.random() < 0.35:
        s += random.choice([".", ".", ",", ":", "?"])
    return s


def ascii_or_hebrew_marks(s):
    """Real documents mostly type ASCII " ' -; sometimes the proper Hebrew marks."""
    if random.random() < 0.1:
        s = re.sub(r'(?<=[א-ת])"(?=[א-ת])', "״", s)
        s = re.sub(r"(?<=[א-ת])'", "׳", s)
    return s


def make_text_mixed(words, ttext, eng, etext):
    r = random.random()
    if r < 0.38:
        toks = random.choice(ttext).split()
        k = random.randint(2, 9)
        i = random.randint(0, max(0, len(toks) - k))
        s = " ".join(toks[i:i + k])
    elif r < 0.60:
        s = ascii_or_hebrew_marks(doc_line(words))
    elif r < 0.80:
        s = ascii_or_hebrew_marks(mixed_line(words, ttext, eng))
    elif r < 0.87:
        s = english_line(eng, etext)
    else:
        s = random.choice([heb_word(words), doc_token(words), number_token(), random.choice(GERSHAYIM),
                           heb_word(words) + " " + heb_word(words), random.choice(LABELS) + ":",
                           eng_term(eng), eng_term(eng), random.choice(PREFIX) + "-" + eng_term(eng)])
    return unicodedata.normalize("NFC", " ".join(s.split()))


def make_text(words, ttext):
    if MIXED:
        return make_text_mixed(words, ttext, _G["eng"], _G["etext"])
    r = random.random()
    if r < 0.50:
        toks = random.choice(ttext).split()
        k = random.randint(2, 9)
        i = random.randint(0, max(0, len(toks) - k))
        s = " ".join(toks[i:i + k])
    elif r < 0.82:
        s = ascii_or_hebrew_marks(doc_line(words))
    else:
        s = random.choice([heb_word(words), doc_token(words), number_token(), random.choice(GERSHAYIM),
                           heb_word(words) + " " + heb_word(words), random.choice(LABELS) + ":"])
    s = unicodedata.normalize("NFC", " ".join(s.split()))
    return s


# ----------------------------------------------------------------- fonts
@lru_cache(maxsize=None)
def cmap(path):
    return frozenset(TTFont(path, fontNumber=0, lazy=True).getBestCmap())


def read_font_list(list_file, font_dirs):
    """Font paths in the order of the list file (first column; '#' starts a comment).
    The order matters: fonts are drawn by position, so another order gives other images."""
    names = []
    for line in open(list_file, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line.split("\t")[0].strip())
    paths, missing = [], []
    for n in names:
        hit = next((Path(d) / n for d in font_dirs if (Path(d) / n).is_file()), None)
        (paths if hit else missing).append(str(hit) if hit else n)
    if missing:
        sys.exit(f"{len(missing)} font(s) from {list_file} not found in {', '.join(map(str, font_dirs))}: "
                 + ", ".join(missing) + "\n(fetch_fonts.sh downloads them)")
    return paths


def font_list(paths):
    out = []
    for p in paths:
        try:
            c = cmap(p)
        except Exception:
            continue
        if all(ord(ch) in c for ch in "אבגםןץ0123456789.,"):
            out.append((p, 0.25 if DISPLAY.search(Path(p).name) else 1.0))
    return out


@lru_cache(maxsize=None)
def caps_latin(path):
    """True if the font draws lowercase Latin like capitals (some Hebrew display fonts):
    a line labelled `clipboard` would show `CLIPBOARD`."""
    c = cmap(path)
    if not all(ord(ch) in c for ch in "xX"):
        return False
    f = ImageFont.truetype(path, 100)
    hx, hX = (f.getbbox(ch)[3] - f.getbbox(ch)[1] for ch in "xX")
    gl = TTFont(path, fontNumber=0, lazy=True).getBestCmap()
    return gl[ord("x")] == gl[ord("X")] or hx >= 0.85 * hX


def pick_font(fonts, text):
    need = {ord(c) for c in text if not c.isspace()}
    lower = any("a" <= c <= "z" for c in text)
    ok = [(p, w) for p, w in fonts if need <= cmap(p) and not (lower and caps_latin(p))]
    if not ok:
        return None
    return random.choices([p for p, _ in ok], weights=[w for _, w in ok])[0]


# ----------------------------------------------------------------- render + degrade
def render(text, font_path, neighbours=("", "")):
    size = random.randint(26, 60)
    font = ImageFont.truetype(font_path, size, layout_engine=ImageFont.Layout.RAQM)
    direction = "rtl" if re.search(r"[֐-׿]", text) else "ltr"
    l, t, r, b = font.getbbox(text, direction=direction)
    padx, pady = random.randint(3, 16), random.randint(2, 10)
    w, h = r - l + 2 * padx, b - t + 2 * pady
    paper = random.choice([255, 255, 255, random.randint(220, 250)])
    img = Image.new("L", (w, h), paper)
    d = ImageDraw.Draw(img)
    ink = random.choice([0, 0, random.randint(20, 90)])
    d.text((padx - l, pady - t), text, font=font, fill=ink, direction=direction)
    if random.random() < 0.30:                      # real crops cut into the lines above / below
        lh = b - t
        for k, nb in enumerate(neighbours):
            if not nb or random.random() < 0.3:
                continue
            nl, nt, nr, nb_ = font.getbbox(nb, direction=direction)
            x = padx - l + (r - l) - (nr - nl) + random.randint(-40, 40)     # right-aligned, jittered
            show = random.uniform(0.15, 0.45) * lh                            # how much of it pokes in
            y = show - nb_ if k == 0 else h - show - nt                       # above: its bottom; below: its top
            d.text((x - nl, y), nb, font=font, fill=ink, direction=direction)
    if random.random() < 0.04:                      # underlined line (headings, links)
        asc, desc = font.getmetrics()
        uy = pady - t + asc + int(desc * random.uniform(0.15, 0.6))
        d.line([(padx, uy), (w - padx, uy)], fill=ink, width=max(1, size // random.randint(12, 20)))
    return img, size


def ink_frac(im):
    a = np.asarray(im, dtype=np.float32)
    return float(((np.median(a) - a) > 60).mean())


def old_print(im):
    """Old newspaper letterpress: uneven ink, broken strokes, spread, then binarised."""
    a = np.asarray(im.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 0.9))), dtype=np.float32)
    a = a + np.random.normal(0, random.uniform(6, 22), a.shape)             # ink density varies
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    thr = random.randint(115, 165)                                           # low: broken, high: bleed
    im = im.point(lambda v: 255 if v > thr else 0)
    if random.random() < 0.25:                                               # spread ink (never thin it)
        im = im.filter(ImageFilter.MinFilter(3))
    return im


def degrade(img, size):
    a = np.asarray(img, dtype=np.float32)
    im = img
    if random.random() < 0.15:
        im = old_print(im)
    if random.random() < 0.35:                      # thin or thicken ink
        t = im.filter(ImageFilter.MaxFilter(3) if random.random() < 0.5 else ImageFilter.MinFilter(3))
        if _G.get("legacy") or ink_frac(t) > 0.5 * ink_frac(im):   # thinning erased light fonts (round 3: 3% blank)
            im = t
    if random.random() < 0.3:
        im = im.rotate(random.uniform(-1.2, 1.2), resample=Image.BICUBIC, expand=True, fillcolor=int(a.max()))
    if random.random() < 0.5:                       # low resolution scan
        f = max(random.uniform(0.35, 0.85), 16 / size)    # never below ~16 px text
        small = im.resize((max(8, int(im.width * f)), max(8, int(im.height * f))), Image.BILINEAR)
        im = small if random.random() < 0.5 else small.resize(im.size, Image.BILINEAR)
    if random.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.1)))
    a = np.asarray(im, dtype=np.float32)
    if random.random() < 0.5:
        a = a + np.random.normal(0, random.uniform(3, 18), a.shape)
    if random.random() < 0.1:                       # speckles (old print)
        m = np.random.random(a.shape) < random.uniform(0.001, 0.01)
        a[m] = np.random.choice([0, 255], size=m.sum())
    if random.random() < 0.15:                      # uneven exposure
        a = a * np.linspace(random.uniform(0.85, 1.0), random.uniform(0.85, 1.0), a.shape[1])[None, :]
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.08:                      # binarized fax/photocopy look
        thr = random.randint(110, 170)
        im = im.point(lambda v: 255 if v > thr else 0)
    if random.random() < 0.4:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=random.randint(25, 90))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return im


# ----------------------------------------------------------------- driver
_G = {}


def _init(langdata, font_paths, mixed=False, clean=False, legacy=False):
    set_profile(mixed)
    langdata = Path(langdata)
    _G["clean"] = clean
    _G["legacy"] = legacy
    _G["words"] = load_words(langdata / "heb.wordlist")
    _G["ttext"] = load_training_text(langdata / "heb.training_text")
    _G["fonts"] = font_list(font_paths)
    if mixed:                                       # langdata_lstm eng files (Apache-2.0)
        _G["eng"] = [w for w in (l.strip() for l in open(langdata / "eng.wordlist", encoding="utf-8"))
                     if w and w.isascii() and w.isalpha()][:150000]
        _G["etext"] = [l.strip() for l in open(langdata / "eng.training_text", encoding="utf-8") if l.strip()]


def _ok(text):
    return bool(text) and set(text) <= ALLOWED and (MIXED or not LATIN.search(text))


def _ok_neighbour(n, fp):
    if _G["legacy"]:                                # round 3 also tried a font for an empty line (uses a random draw)
        return set(n) <= ALLOWED and not LATIN.search(n) and pick_font([(fp, 1.0)], n)
    return _ok(n) and pick_font([(fp, 1.0)], n)


def _one(args):
    idx, seed, out = args
    random.seed(seed); np.random.seed(seed % (2 ** 32))
    for _ in range(20):
        text = make_text(_G["words"], _G["ttext"])
        if not _ok(text):
            continue
        fp = pick_font(_G["fonts"], text)
        if fp:
            break
    else:
        return None
    nbs = ("", "") if _G["clean"] else tuple(make_text(_G["words"], _G["ttext"]) for _ in range(2))
    nbs = tuple(n if _ok_neighbour(n, fp) else "" for n in nbs)
    img, size = render(text, fp, nbs)
    if _G["legacy"]:
        img = degrade(img, size)
    elif not _G["clean"]:
        clean_ink = ink_frac(img)
        for _ in range(3):                          # never train on a line the degradation wiped out
            out_img = degrade(img, size)
            if ink_frac(out_img) > 0.25 * clean_ink:
                break
        else:
            out_img = img
        img = out_img
    base = Path(out) / f"{idx:06d}"
    img.save(f"{base}.png")
    Path(f"{base}.gt.txt").write_text(text + "\n", encoding="utf-8")
    return Path(fp).name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--langdata", required=True, help="folder with heb.wordlist, heb.training_text (+ eng.* for --mixed)")
    ap.add_argument("--fonts", required=True, help="font list (fonts.txt); its order matters")
    ap.add_argument("--font-dir", action="append", required=True, help="folder holding the listed fonts (repeatable)")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--mixed", action="store_true", help="round 4: Hebrew with English inside, English lines")
    ap.add_argument("--clean", action="store_true", help="no degradation, no neighbour fragments")
    ap.add_argument("--legacy-r3", action="store_true", help="round 3's generator exactly (heb_rivok_lite)")
    a = ap.parse_args()
    if a.legacy_r3 and (a.mixed or a.clean):
        ap.error("--legacy-r3 reproduces round 3 and cannot be combined with --mixed or --clean")
    assert features.check("raqm"), "Pillow needs raqm for Hebrew shaping"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    font_paths = read_font_list(a.fonts, a.font_dir)
    jobs = [(i, a.seed * 1000003 + i, str(out)) for i in range(a.start, a.start + a.n)]
    with Pool(a.jobs, _init, (a.langdata, font_paths, a.mixed, a.clean, a.legacy_r3)) as pool:
        used = [f for f in pool.imap_unordered(_one, jobs, chunksize=64) if f]
    from collections import Counter
    print(f"{len(used)} lines written to {out}; fonts used: {len(set(used))} of {len(font_list(font_paths))}")
    for f, n in Counter(used).most_common(8):
        print(f"  {n:6d} {f}")


if __name__ == "__main__":
    main()
