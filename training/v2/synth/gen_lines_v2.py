#!/usr/bin/env python3
"""heb_rivok 2.0 synthetic line images (public copy of the generator used for 2.0; see
training/v2/README.md and train_v2.sh for the exact runs), built on the 1.0 (round-4) generator.

gen_lines.py --mixed (heb_rivok's text mix, fonts, render and damage) is imported unchanged and
used as the "replay" stream. v2 adds four streams and six output characters:

  stream     default share  text                                         fonts
  replay     25             gen_lines --mixed (v1 mix: training-text      v1 choice (display faces
                            slices, documents, Hebrew/English, English)   at 1/4 weight)
  vowelled   20             passages from --vowelled sources (Tanakh,     fonts that draw every mark
                            Mishnah, poetry, Dicta-vowelled modern text)  in the line correctly (85%)
  confusion  10             old-type look-alike pairs: real-word minimal  serif / old-type faces,
                            pairs (דבר/רבר-style) and words rich in       letterpress damage 50%
                            ד ר ב כ ז ו ל נ ג ם ס ה ח ת
  fonts      10             unvowelled Hebrew lines                        every face, display faces
                                                                          at full weight
  negatives   5             Hebrew lines rich in letters that look like   display faces x3
                            digits / Latin (ו ן ס ם ט ד ז ש צ…), no digits;
                            20% with real digits next to look-alikes
Added 2026-11-12 for the 2.0 training run:
  siddur      5             vowelled windows rich in kaf/resh/bet/pe,     siddur-type serif faces
                            tet/shin, zayin/vav and vav/yod under holam/  that draw the marks
                            hiriq/shuruk (common letter errors)
  signs       5             1-3 short words: sign / menu words (קפה,     display and bold sans
                            בירה, יציאה…) or short words, sometimes a price  faces x3
  docnum      5             Hebrew + numbers document lines: label:       all faces
                            value pairs, amounts, dates, IDs, quantities
  negatives                 + column rules, neighbour-column slivers and specks drawn into the
                            image and NOT in the label; 15% of lines carry a real "|" (in the label)
  all streams               15% faded: low contrast, uneven light (low-contrast photos)
The other 25% of the v2 mix is real Wikisource lines, not made here.

New characters (2026-11-08): meteg U+05BD, hataf segol U+05B1, hataf qamats U+05B3,
qubuts U+05BB, holam haser for vav U+05BA, qamats qatan U+05C7. No cantillation.

Image and label always agree. Before rendering, a line is adapted to its font
(font_caps.json, made from our font coverage tests):
  - a mark the font does not draw correctly is removed from text and label;
  - qamats qatan / holam haser for vav become qamats / holam where the font draws them
    identically (no visible difference to learn).
Source cleaning: cantillation U+0591–05AF and the extraordinary dots U+05C4/05C5 removed,
sof pasuq ׃ → ":", paseq ׀ and nun hafukha removed, OSHB morpheme slashes removed, NFC.

Damage: 40% heavy (gen_lines.degrade, letterpress first in a third of them), 60% light (mild
blur / noise / JPEG / slight downscale). Vowelled lines: heavy only 20%, and text is never
scaled below ~22 px so marks stay visible.

Output: OUT/NNNNNN.png + OUT/NNNNNN.gt.txt (logical order, NFC) + OUT/manifest.tsv
(idx, stream, source, font, damage).

    python3 -I gen_lines_v2.py OUT --langdata LANGDATA --fonts FONTS --n N --seed S [--start I]
        [--jobs 4] [--mix ...] [--vowelled NAME=PATH:WEIGHT ...] [--only I,J,...]
LANGDATA holds heb.wordlist, heb.training_text, eng.wordlist and eng.training_text (fetch_inputs.sh);
FONTS holds the font files of fonts_v2.tsv (fetch_v2_sources.py --only fonts, archives unpacked).
--only renders just those line numbers of the run (the stream draws are replayed for every line
from --start, so each listed line comes out as in the full run).
Without --vowelled, the vowelled stream uses the vowelled lines of langdata heb.training_text
and a few hand-written lines with the new marks (smoke tests only).
"""
import argparse
import io
import json
import random
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # training/gen_lines.py, the 1.0 generator
import gen_lines as G  # noqa: E402

HERE = Path(__file__).resolve().parent
FONT_LIST = HERE / "fonts_v2.tsv"        # the 2.0 run's font order and weights
FONT_CAPS = HERE / "font_caps.json"      # marks each font draws correctly
NEW_MARKS = "ֱֳׇֺֻֽ"
OLD_MARKS = "ְֲִֵֶַָֹּֿׁׂ"
MARK_NAMES = {"ְ": "sheva", "ַ": "patah", "ָ": "qamats", "ּ": "dagesh", "ִ": "hiriq",
              "ֶ": "segol", "ֹ": "holam", "ֵ": "tsere", "ׁ": "shin dot", "ׂ": "sin dot",
              "ֲ": "hataf patah", "ֿ": "rafe", "ֽ": "meteg", "ֱ": "hataf segol",
              "ֳ": "hataf qamats", "ֻ": "qubuts", "ֺ": "holam haser for vav", "ׇ": "qamats qatan"}
FOLD = {"ׇ": "ָ", "ֺ": "ֹ"}          # look-alike when a font draws them the same
STREAMS = ("replay", "vowelled", "confusion", "fonts", "negatives", "siddur", "signs", "docnum", "latin", "rare")
DEFAULT_MIX = "replay=20,vowelled=20,confusion=5,siddur=5,fonts=8,negatives=7,signs=5,docnum=5"
VOWELLED_STREAMS = {"vowelled", "siddur", "rare"}
RARE_MARKS = "\u05bb\u05bd\u05b3\u05ba\u05c7"   # qubuts, meteg, hataf qamats, holam haser, qamats qatan:
                                                   # the new marks v2's main run learned slowly (2026-11-13)
FADED_P = 0.15
SIDDUR_LETTERS = set("כרבפטשזך")
SIDDUR_THIN = re.compile("[וי][\u05b4\u05b9\u05bc]|[\u05b4]י")    # vav/yod with holam/hiriq/shuruk, hiriq + yod
SIGN_FONT = re.compile(r"Heebo|Alef|Noto ?Sans|Rubik|Assistant|Secular|Suez|Varela|Open ?Sans|Arimo|Liberation ?Sans|"
                       r"DejaVu ?Sans|FreeSans|Karantina|Amatic|Bellefair|Bold|Black|Heavy|Arial|Simple|Miriam", re.I)
SIGN_WORDS = ("קפה בירה יין תה מים מיץ לחם פיצה סלט מרק עוגה גלידה שווארמה פלאפל חומוס סושי המבורגר "
              "יציאה כניסה מעלית מדרגות שירותים חניה חנייה אין חניה סגור פתוח מבצע מבצעים הנחה חדש "
              "מאפייה מסעדה בית קפה פיצרייה מכולת סופר מרקט קופה קופות משלוחים איסוף עצמי "
              "קבלה מודיעין משרד מרפאה בית מרקחת מספרה ספרייה בית ספר גן ילדים תחנה אוטובוס רכבת "
              "שבת שלום ברוכים הבאים תודה רבה היכנסו זהירות אסור לעשן עשן לא לגעת שקט נא "
              "רחוב שדרות כיכר פינת שוק קניון בנק דואר פתוח 24/7").split()

OLD_TYPE = re.compile(r"Frank|David|Tinos|Serif|Bellefair|Drugulin|Hadasim|Keter|Shofar|Taamey|Ezra|Cardo|"
                      r"Libertinus|Bona|Times|Miriam|Nachlieli|Yehuda|Stam", re.I)
CONF_PAIRS = [("ד", "ר"), ("ב", "כ"), ("ז", "ו"), ("ר", "ל"), ("נ", "ג"), ("ם", "ס"), ("ה", "ח"), ("ח", "ת"), ("ה", "ת")]
CONF_LETTERS = set("דרבכזולנגםסהחת")
DIGITLIKE = set("וןסםטדזשצי")                           # read as 1 l I 0 O 6 v 7 t w y '
DIGIT_NEIGHBOURS = {"ו": "1", "ן": "1", "ס": "0", "ם": "0", "ט": "6", "ד": "7", "ז": "1"}
# hand-written lines with the six new marks (Biblical / standard pointing), for smoke tests only
BUILTIN_VOWELLED = [
    "כׇּל־הָעָם עָנוּ יַחְדָּו וַיֹּאמְרוּ", "רֵאשִׁית חׇכְמָה יִרְאַת ה׳", "חֳלִי וּמַכְאוֹב בַּצׇּהֳרַיִם",
    "קֻמִּי אוֹרִי כִּי בָא אוֹרֵךְ", "נֹשֵׂא עָוֺן וָפֶשַׁע וְחַטָּאָה", "הָֽאָדָם הָֽרִאשׁוֹן בַּגָּן",
    "וַיֹּאמֶר אֱלֹהִים יְהִי אוֹר וַיְהִי־אוֹר", "אׇזְנַיִם לָהֶם וְלֹא יִשְׁמָעוּ", "עֳנִי וְעֹשֶׁר אַל־תִּתֶּן־לִי",
    "בְּנֵי־אֱלֹהִים אֱמֶת וֶאֱמוּנָה", "מִצְוֺת עֲשֵׂה וְלֹא תַעֲשֶׂה", "שֻׁלְחָן עָרוּךְ וְקׇדְשֵׁי קֳדָשִׁים",
    "וַֽיְהִי בַּיָּמִים הָהֵם", "תׇּכְנִית לִמּוּדִים חֳדָשִׁית", "אֱנוֹשׁ כֶּחָצִיר יָמָיו",
]
_V = {}


# ----------------------------------------------------------------- text cleaning
def clean_vowelled(s):
    s = s.translate(G.BIDI_CTRL).replace("/", "")
    s = re.sub(r"[֑-ׅ֯ׄ׆׀]", "", s)       # cantillation, dots, nun hafukha, paseq
    s = s.replace("׃", ":")                                          # sof pasuq
    s = unicodedata.normalize("NFC", " ".join(s.split()))
    return s


def window(passage, lo=2, hi=9):
    toks = passage.split()
    if random.random() < 0.1:
        lo, hi = 1, 3
    k = random.randint(lo, hi)
    i = random.randint(0, max(0, len(toks) - k))
    return " ".join(toks[i:i + k])


# ----------------------------------------------------------------- font capabilities
def load_capabilities(path):
    """{font file name: set of marks it draws correctly} (made from our font coverage tests)."""
    return {k: set(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}


def adapt(text, font):
    """Make the label match what the font can draw: fold look-alikes, drop undrawable marks."""
    ok = _V["caps"].get(Path(font).name)
    if ok is None:                                       # font not in the coverage table: strip new marks
        ok = set(OLD_MARKS)
    out = []
    for ch in text:
        if ch in MARK_NAMES and ch not in ok:
            if ch in FOLD and FOLD[ch] in ok:
                out.append(FOLD[ch])
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def font_pool(stream):
    fonts = _V["fonts"]
    rashi = re.compile(r"Rashi", re.I)                    # Rashi script: out of v2 scope beyond v1's 1/4 weight
    if stream == "confusion":
        return [(p, 0.0 if rashi.search(Path(p).name) else 3.0 if OLD_TYPE.search(Path(p).name) else 0.3) for p, _ in fonts]
    if stream == "fonts":
        return [(p, 0.25 if rashi.search(Path(p).name) else 1.0) for p, _ in fonts]
    if stream == "negatives":
        return [(p, 0.25 if rashi.search(Path(p).name) else 3.0 if G.DISPLAY.search(Path(p).name) else 1.0) for p, _ in fonts]
    if stream == "siddur":
        return [(p, 0.0 if rashi.search(Path(p).name) else 3.0 if OLD_TYPE.search(Path(p).name) else 0.2) for p, _ in fonts]
    if stream == "signs":
        return [(p, 0.0 if rashi.search(Path(p).name) else 3.0 if (G.DISPLAY.search(Path(p).name) or SIGN_FONT.search(Path(p).name)) else 0.5)
                for p, _ in fonts]
    return fonts


def pick(stream, text):
    """(font, adapted text) or (None, None)."""
    if stream in ("replay", "latin"):
        fp = G.pick_font(G._G["fonts"], text)
        return fp, text
    pool = [(p, w) for p, w in font_pool(stream) if w > 0]
    marks = {c for c in text if c in MARK_NAMES}
    if stream in VOWELLED_STREAMS:
        native = [(p, w) for p, w in pool if marks <= _V["caps"].get(Path(p).name, set())]
        if native and random.random() < 0.85:
            pool = native
        else:                                            # vowelled lines only in fonts with all 12 old marks
            pool = [(p, w) for p, w in pool if set(OLD_MARKS) <= _V["caps"].get(Path(p).name, set())] or pool
    for _ in range(6):
        fp = G.pick_font(pool, re.sub(r"[֑-ׇ]", "", text))
        if not fp:
            return None, None
        t = adapt(text, fp) if marks else text
        if t and {ord(c) for c in t if not c.isspace()} <= G.cmap(fp):
            return fp, t
    return None, None


# ----------------------------------------------------------------- streams
def s_replay():
    return G.make_text(G._G["words"], G._G["ttext"]), "v1"


def s_latin():
    """v1's lines without Hebrew (English, numbers, codes): round 4 kept them so English is not
    forgotten. The Hebrew check in _one dropped them from replay (found 2026-11-12, mid-run)."""
    for _ in range(60):
        t = G.make_text(G._G["words"], G._G["ttext"])
        if t and not re.search(r"[א-ת]", t):
            return t, "v1_latin"
    return "", "v1_latin"


VOWELLED_WORD = re.compile(r"[\u05b0-\u05bd\u05bf\u05c1\u05c2\u05c7]")


def s_vowelled():
    """A window of a vowelled passage in which most words carry marks (sources such as Tesseract's
    training text mix vowelled and plain words in one line)."""
    srcs = _V["vowelled"]
    name, lines, _ = random.choices(srcs, weights=[w for _, _, w in srcs])[0]
    best, best_share = "", -1.0
    for _ in range(12):
        w = window(random.choice(lines))
        toks = [t for t in w.split() if re.search(r"[א-ת]", t)]
        share = sum(bool(VOWELLED_WORD.search(t)) for t in toks) / max(1, len(toks))
        if share > best_share:
            best, best_share = w, share
        if share >= 0.6:
            break
    return best, name


def conf_word():
    if random.random() < 0.5 and _V["pairs"]:
        a, b = random.choice(_V["pairs"])
        return random.choice([a, b, f"{a} {b}", f"{b} {a}"])
    return random.choice(_V["conf_words"])


def s_confusion():
    toks = []
    while len(" ".join(toks).split()) < random.randint(3, 8):
        toks.append(conf_word())
    s = " ".join(toks)
    if random.random() < 0.3:
        s += random.choice([".", ",", ":", ";"])
    return s, "pairs"


def s_fonts():
    if random.random() < 0.5:
        toks = random.choice(G._G["ttext"]).split()
        k = random.randint(2, 8)
        i = random.randint(0, max(0, len(toks) - k))
        s = " ".join(G.MARKS.sub("", t) or t for t in toks[i:i + k])
    else:
        s = G.doc_line(G._G["words"])
    return unicodedata.normalize("NFC", " ".join(s.split())), "plain"


def s_negatives():
    toks = [random.choice(_V["neg_words"]) for _ in range(random.randint(2, 7))]
    if random.random() < 0.15 and len(toks) >= 2:          # a real vertical bar, labelled (tables, menus)
        toks.insert(random.randint(1, len(toks) - 1), "|")
    if random.random() < 0.2:                            # contrast: a real digit next to its look-alike
        i = random.randrange(len(toks))
        w = toks[i]
        cands = [c for c in w if c in DIGIT_NEIGHBOURS]
        if cands:
            c = random.choice(cands)
            toks.insert(i + 1, DIGIT_NEIGHBOURS[c] * random.randint(1, 3))
    return " ".join(toks), "lookalike"


def s_siddur():
    """A vowelled window dense in the letters OCR engines confuse in prayer-book type."""
    srcs = _V["vowelled"]
    name, lines, _ = random.choices(srcs, weights=[w for _, _, w in srcs])[0]
    best, best_score = "", -1.0
    for _ in range(16):
        w = window(random.choice(lines), 2, 7)
        toks = [t for t in w.split() if re.search(r"[א-ת]", t)]
        if not toks:
            continue
        share = sum(bool(VOWELLED_WORD.search(t)) for t in toks) / len(toks)
        letters = [c for c in w if "א" <= c <= "ת"]
        score = (sum(c in SIDDUR_LETTERS for c in letters) / max(1, len(letters))
                 + 0.5 * len(SIDDUR_THIN.findall(w)) / max(1, len(toks)))
        if share >= 0.6 and score > best_score:
            best, best_score = w, score
    return best, "siddur:" + name


def s_rare():
    """A short vowelled window around a word with one of the slow-to-learn new marks. Rarer marks
    are drawn more often: each mark has its own pool of source words."""
    m = random.choice(RARE_MARKS)
    pool = _V["rare"].get(m)
    if not pool:
        return "", "rare"
    line, k = random.choice(pool)
    toks = line.split()
    lo = max(0, k - random.randint(0, 3))
    return " ".join(toks[lo:lo + random.randint(2, 6)]), "rare:" + MARK_NAMES[m]


def s_signs():
    toks = []
    for _ in range(random.choices([1, 2, 3], [0.45, 0.4, 0.15])[0]):
        if random.random() < 0.6:
            toks.append(random.choice(SIGN_WORDS))
        else:
            w = random.choice(_V["short_words"])
            toks.append(w)
    s = " ".join(toks)
    r = random.random()
    if r < 0.12:
        s += " " + random.choice([f"{random.randint(5, 60)} ₪", f"₪{random.randint(5, 60)}",
                                  f"{random.randint(5, 60)}.{random.choice(['90', '50', '00'])}"])
    elif r < 0.18:
        s += random.choice(["!", ".", ":"])
    return s, "signs"


def s_docnum():
    W = G._G["words"]
    kind = random.random()
    if kind < 0.45:                                       # 1-3 label: value pairs
        parts = []
        for _ in range(random.choices([1, 2, 3], [0.5, 0.35, 0.15])[0]):
            lab = random.choice(G.LABELS + ["סה\"כ", "מע\"מ", "ח.פ.", "ע.מ.", "מס' חשבונית", "מס' הזמנה", "אסמכתא",
                                            "קבלה מס'", "תאריך הפקה", "תאריך אספקה", "לכבוד", "ת.ז.", "מספר עוסק"])
            val = random.choice([G.money(), G.date(), G.phone(), G.r_int(100000, 999999999), G.number_token(),
                                 G.r_int(1, 99999)])
            parts.append(f"{lab}{random.choice([':', ': ', ' :', ''])} {val}".replace(":  ", ": "))
        return random.choice(["  ", " ", ", ", " | " if random.random() < 0.2 else " "]).join(parts), "docnum"
    if kind < 0.7:                                        # item line: description, quantity, price, total
        desc = " ".join(G.heb_word(W) for _ in range(random.randint(1, 4)))
        q = random.randint(1, 24)
        price = f"{random.randint(1, 999)}.{random.randint(0, 99):02d}"
        return random.choice([f"{desc} {q} x {price}", f"{desc} {price}", f"{q} {desc} {price} ₪",
                              f"{desc} ({q} יח') {G.money()}"]), "docnum"
    toks = [G.heb_word(W) if random.random() < 0.55 else G.number_token() for _ in range(random.randint(3, 8))]
    if not any(re.search(r"\d", t) for t in toks):
        toks.insert(random.randrange(len(toks) + 1), G.number_token())
    return " ".join(toks), "docnum"


MAKERS = {"replay": s_replay, "vowelled": s_vowelled, "confusion": s_confusion, "fonts": s_fonts,
          "negatives": s_negatives, "siddur": s_siddur, "signs": s_signs, "docnum": s_docnum, "latin": s_latin,
          "rare": s_rare}


# ----------------------------------------------------------------- damage
def light(img):
    im = img
    if random.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.7)))
    if random.random() < 0.4:
        f = random.uniform(0.7, 0.95)
        im = im.resize((max(8, int(im.width * f)), max(8, int(im.height * f))), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    if random.random() < 0.5:
        a = a + np.random.normal(0, random.uniform(2, 8), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.4:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=random.randint(60, 95))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return im


def heavy(img, size, stream):
    im = img
    if random.random() < (0.5 if stream == "confusion" else 0.33):
        im = G.old_print(im)
    # vowelled lines with marks use heavy_v (capped) and the mark check, not this
    return G.degrade(im, size)


def edge_noise(img):
    """Things next to a line that are not text (labelled as nothing): a column rule, a sliver of a
    neighbouring column's letters, specks. Phase 4 found heb_rivok reads column rules as | and
    letters; OCR tends to invent ו/י/ה from specks."""
    a = np.asarray(img, dtype=np.float32)
    paper = float(np.median(a))
    ink = float(min(a.min(), paper - 80))
    h, w = a.shape
    r = random.random()
    if r < 0.4:                                            # column rule beside the text
        gap, width = random.randint(3, 22), random.randint(1, 3)
        new = np.full((h + random.randint(0, 8), w + gap + width + random.randint(0, 6)), paper, np.float32)
        right = random.random() < 0.5
        x0 = 0 if right else new.shape[1] - w
        new[:h, x0:x0 + w] = a
        xr = (w + gap) if right else (new.shape[1] - w - gap - width)
        new[:, xr:xr + width] = ink + random.uniform(0, 40)
        a = new
    elif r < 0.7:                                          # sliver of the neighbouring column's letters
        sw = random.randint(3, max(4, min(14, w // 8)))
        x = random.randint(0, max(0, w - sw))
        strip = a[:, x:x + sw]
        gap = random.randint(2, 14)
        pad = np.full((h, gap), paper, np.float32)
        a = np.concatenate([a, pad, strip], 1) if random.random() < 0.5 else np.concatenate([strip, pad, a], 1)
    if random.random() < 0.5:                              # specks, some close to the letters
        n = random.randint(3, 25)
        ys, xs = np.random.randint(0, a.shape[0], n), np.random.randint(0, a.shape[1], n)
        for y, x in zip(ys, xs):
            rad = random.choice([0, 0, 1, 1, 2])
            val = random.uniform(ink, paper - 40)
            a[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1] = val
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def faded(img, keep_marks=False):
    """Faded print and uneven light: ink pulled towards the paper, a light gradient across the
    line, a little noise. Ink always stays >= 45 levels darker than the paper."""
    a = np.asarray(img, dtype=np.float32)
    paper = float(np.percentile(a, 90))
    dark = float(a.min())
    span = max(1.0, paper - dark)
    k = max(random.uniform(0.25, 0.6), 45.0 / span)
    a = paper - (paper - a) * k
    a = a + random.uniform(0, max(0.0, 250 - paper))       # paper may also brighten
    h, w = a.shape
    if random.random() < 0.7:                              # uneven light: a ramp or a soft spot
        if random.random() < 0.6:
            ramp = np.linspace(1.0, 1.0 - random.uniform(0.08, 0.3), w if random.random() < 0.7 else h)
            field = ramp[None, :] if ramp.size == w else ramp[:, None]
        else:
            yy, xx = np.mgrid[0:h, 0:w]
            cy, cx = random.uniform(0, h), random.uniform(0, w)
            field = 1.0 - random.uniform(0.08, 0.25) * np.exp(-(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2) * 2)
        a = a * field
    a = a + np.random.normal(0, random.uniform(0.5, 2.0 if keep_marks else 4.0), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------- vowelled lines: size floor, capped damage, mark check
# Damage must never erase a mark that the label keeps (that would teach the model to output marks
# it cannot see, i.e. to invent marks on plain text). So a vowelled line is rendered twice on one
# canvas, with and without its marks; the same damage (same seed) is applied to both; every mark
# must still differ visibly between the two. Otherwise: heavy -> light damage; light fails -> drop.
BODY_MIN = 30        # px: letter body (ink height of ב) at render time
BODY_FLOOR = 28      # px: damage never scales the letter body below this
VIS_TAU = 0.5        # every mark's peak contrast difference after damage, as a share of the
                     # line's ink contrast, must reach this. Calibrated by eye on 4,499 marks
                     # (our calibration sheet): < 0.3 gone, 0.3-0.45 a faint smudge,
                     # 0.45-0.6 faint but visible, >= 0.6 clear


@lru_cache(maxsize=None)
def body_ratio(fp):
    l, t, r, b = ImageFont.truetype(fp, 100, layout_engine=ImageFont.Layout.RAQM).getbbox("ב", direction="rtl")
    return max(0.2, (b - t) / 100)


def strip_marks(text):
    return "".join(c for c in text if c not in MARK_NAMES)


def render_pair(text, fp, body_min=BODY_MIN):
    """(with marks, without marks, letter body px, ink contrast): same font, size, canvas, origin.
    Rotation, if any, is applied here to both, so the damage that follows is only a scale."""
    ratio = body_ratio(fp)
    smin = int(np.ceil(body_min / ratio))
    size = random.randint(smin, int(smin * 1.5))
    font = ImageFont.truetype(fp, size, layout_engine=ImageFont.Layout.RAQM)
    l, t, r, b = font.getbbox(text, direction="rtl")
    padx, pady = random.randint(3, 16), random.randint(4, 12)
    w, h = r - l + 2 * padx, b - t + 2 * pady
    paper = random.choice([255, 255, 255, random.randint(220, 250)])
    ink = random.choice([0, 0, random.randint(20, 90)])
    pair = []
    for s in (text, strip_marks(text)):
        im = Image.new("L", (w, h), paper)
        ImageDraw.Draw(im).text((padx - l, pady - t), s, font=font, fill=ink, direction="rtl")
        pair.append(im)
    if random.random() < 0.3:
        ang = random.uniform(-1.2, 1.2)
        pair = [im.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=paper) for im in pair]
    return pair[0], pair[1], ratio * size, paper - ink


def _scale(im, body_px, lo, hi, floor=BODY_FLOOR):
    fmin = max(lo, floor / body_px)
    if fmin >= hi:
        return im
    f = random.uniform(fmin, hi)
    small = im.resize((max(8, round(im.width * f)), max(8, round(im.height * f))), Image.BILINEAR)
    return small if random.random() < 0.5 else small.resize(im.size, Image.BILINEAR)


def _jpeg(im, lo, hi):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=random.randint(lo, hi))
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def light_v(img, body_px):
    im = img
    if random.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.6)))
    if random.random() < 0.4:
        im = _scale(im, body_px, 0.7, 0.95)
    a = np.asarray(im, dtype=np.float32)
    if random.random() < 0.5:
        a = a + np.random.normal(0, random.uniform(2, 8), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return _jpeg(im, 60, 95) if random.random() < 0.4 else im


def heavy_v(img, body_px, wide=False):
    """Heavy damage for vowelled lines, capped to keep 2-px dots: no ink thinning (a 3x3 max filter
    erases them), blur <= 0.8, noise <= 12, few speckles, letter body >= BODY_FLOOR px.
    wide=True opens the ranges (thinning, blur 1.4, noise 22, body 16 px): calibration only."""
    bmax, nmax, slo, floor = (1.4, 22, 0.35, 16) if wide else (0.8, 12, 0.5, BODY_FLOOR)
    im = img
    r = random.random()
    if r < 0.25:                                         # letterpress: binarised, uneven ink
        a = np.asarray(im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9 if wide else 0.6))), dtype=np.float32)
        a = a + np.random.normal(0, random.uniform(4, nmax), a.shape)
        thr = random.randint(120, 165 if wide else 150)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).point(lambda v: 255 if v > thr else 0)
        if random.random() < 0.15:
            im = im.filter(ImageFilter.MinFilter(3))     # ink spread
    elif r < 0.45:
        im = im.filter(ImageFilter.MinFilter(3))         # thicker ink
    elif wide and r < 0.65:
        im = im.filter(ImageFilter.MaxFilter(3))         # thinner ink
    if random.random() < 0.6:
        im = _scale(im, body_px, slo, 0.9, floor)
    if random.random() < 0.6:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, bmax)))
    a = np.asarray(im, dtype=np.float32)
    if random.random() < 0.6:
        a = a + np.random.normal(0, random.uniform(3, nmax), a.shape)
    if random.random() < 0.05:                           # sparse speckles (dense ones look like marks)
        m = np.random.random(a.shape) < random.uniform(0.0005, 0.002)
        a[m] = np.random.choice([0, 255], size=m.sum())
    if random.random() < 0.15:
        a = a * np.linspace(random.uniform(0.85, 1.0), random.uniform(0.85, 1.0), a.shape[1])[None, :]
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.05:
        thr = random.randint(115, 155)
        im = im.point(lambda v: 255 if v > thr else 0)
    return _jpeg(im, 35, 90) if random.random() < 0.5 else im


def damage_pair(m, p, fn, seed):
    out = []
    for im in (m, p):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        out.append(fn(im))
    return out


def components(mask):
    """8-connected components of a boolean mask, as (N, 2) arrays of (y, x)."""
    lab = np.zeros(mask.shape, np.int32)
    H, W = mask.shape
    comps = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        lab[y0, x0] = len(comps) + 1
        stack, pts = [(y0, x0)], []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for yy in (y - 1, y, y + 1):
                for xx in (x - 1, x, x + 1):
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = len(comps) + 1
                        stack.append((yy, xx))
        comps.append(np.array(pts))
    return comps


def mark_scores(m, p, md, pd, contrast, with_boxes=False):
    """Per mark: peak contrast difference after damage, as a share of the ink contrast, in the
    mark's box (+1 px). Marks = connected components of the ink that the with-marks render has and
    the without-marks render lacks, at least 2 px outside the plain letters: some fonts draw a
    letter slightly differently with its marks (another shin glyph, a 1-px shift), and those
    edge slivers are not marks (they made the check drop lines whose marks were plainly visible)."""
    mi, pi = np.asarray(m, np.int16), np.asarray(p, np.int16)
    letters = (int(pi.max()) - pi) > 0.3 * contrast
    near = np.asarray(Image.fromarray(letters.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0
    C = (np.abs(mi - pi) > 0.5 * contrast) & ~near
    comps = [c for c in components(C) if len(c) >= 3]
    diff = np.abs(np.asarray(md, np.float32) - np.asarray(pd, np.float32))
    H, W = diff.shape
    sy, sx = H / C.shape[0], W / C.shape[1]
    out = []
    for c in comps:
        (y0, x0), (y1, x1) = c.min(0), c.max(0)
        Y0, Y1 = max(0, int(y0 * sy) - 1), min(H, int(np.ceil((y1 + 1) * sy)) + 1)
        X0, X1 = max(0, int(x0 * sx) - 1), min(W, int(np.ceil((x1 + 1) * sx)) + 1)
        s = float(diff[Y0:Y1, X0:X1].max()) / max(1, contrast)
        out.append((s, (X0, Y0, X1, Y1)) if with_boxes else s)
    return out


def vowelled_image(text, fp):
    """(image, damage, check) for a vowelled line; check is pass / fallback (heavy failed, light
    kept every mark) / dropped (light lost a mark too: caller picks another line)."""
    m, p, body_px, contrast = render_pair(text, fp)
    if _V["clean"]:
        return m, "none", "n/a"
    modes = ["heavy", "light"] if random.random() < 0.2 else ["light"]
    clean_ink = G.ink_frac(m)
    for k, mode in enumerate(modes):
        fn = (lambda im: heavy_v(im, body_px)) if mode == "heavy" else (lambda im: light_v(im, body_px))
        md, pd = damage_pair(m, p, fn, random.randrange(2 ** 31))
        scores = mark_scores(m, p, md, pd, contrast)
        if (not scores or min(scores) >= VIS_TAU) and G.ink_frac(md) > 0.25 * clean_ink:
            return md, mode, "pass" if k == 0 else "fallback"
    return None, "none", "dropped"


# ----------------------------------------------------------------- driver
def _font_rows(cfg):
    lines = Path(cfg["font_list"]).read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]
    out = [(Path(cfg["fonts"]) / r["file"], float(r["weight"]), r["in_1.0_list"] == "1") for r in rows]
    missing = [str(p) for p, _, _ in out if not p.exists()]
    if missing:
        sys.exit(f"{len(missing)} fonts missing from {cfg['fonts']}, e.g. {missing[:3]}")
    return out


def _init(cfg):
    fonts = _font_rows(cfg)
    G._init(cfg["langdata"], [str(p) for p, _, v1 in fonts if v1], True, cfg["clean"])
    G._G["fonts"] = [(str(p), w) for p, w, v1 in fonts if v1]     # exactly as in the 2.0 run
    allowed = set(G.BASE + G.ADDED + G.MIXED_EXTRA + NEW_MARKS + " ")
    G.ALLOWED = allowed
    G.UNSUPPORTED_MARKS = re.compile("[" + "".join(chr(c) for c in range(0x0591, 0x05C8) if chr(c) not in allowed) + "]")
    _V["clean"] = cfg["clean"]
    _V["caps"] = load_capabilities(cfg["caps"])
    _V["fonts"] = [(str(p), w) for p, w, _ in fonts]
    # vowelled sources
    srcs = []
    for name, path, w in cfg["vowelled"]:
        lines = [clean_vowelled(l) for l in open(path, encoding="utf-8") if l.strip()]
        srcs.append((name, [l for l in lines if l], w))
    if not srcs:
        tt = [clean_vowelled(l) for l in open(cfg["langdata"] / "heb.training_text", encoding="utf-8")]
        tt = [l for l in tt if len(re.findall(r"[ְ-ׇֽׁׂ]", l)) >= 5 and not G.LATIN.search(l)]
        srcs = [("training_text_vowelled", tt, 4.0), ("builtin_new_marks", [clean_vowelled(l) for l in BUILTIN_VOWELLED], 1.0)]
    _V["vowelled"] = srcs
    rare = {m: [] for m in RARE_MARKS}                   # (line, index of a token carrying the mark)
    for _, lines, _ in srcs:
        for line in lines:
            for k, t in enumerate(line.split()):
                for m in RARE_MARKS:
                    if m in t and len(rare[m]) < 200000:
                        rare[m].append((line, k))
    _V["rare"] = rare
    # confusion and look-alike word pools (unvowelled words from the wordlist)
    words = [w for w in (G.MARKS.sub("", w) for w in G._G["words"][:100000]) if re.fullmatch(r"[א-ת]{2,}", w)]
    wset = set(words)
    pairs = set()
    for w in words[:60000]:
        for i, ch in enumerate(w):
            for a, b in CONF_PAIRS:
                o = b if ch == a else a if ch == b else None
                if o:
                    w2 = w[:i] + o + w[i + 1:]
                    if w2 in wset and w2 != w:
                        pairs.add(tuple(sorted((w, w2))))
    _V["pairs"] = sorted(pairs)
    _V["conf_words"] = [w for w in words if sum(c in CONF_LETTERS for c in w) >= max(2, len(w) // 2)]
    _V["neg_words"] = [w for w in words if sum(c in DIGITLIKE for c in w) >= max(2, (len(w) + 1) // 2)]
    _V["short_words"] = [w for w in words[:20000] if 2 <= len(w) <= 5]


def _one(args):
    idx, seed, out, stream = args
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    drops = 0
    for _ in range(30):
        text, src = MAKERS[stream]()
        text = unicodedata.normalize("NFC", " ".join(text.split()))
        if (not text or not set(text) <= G.ALLOWED or (stream != "latin" and not re.search(r"[א-ת]", text))
                or (stream not in ("replay", "docnum", "latin") and G.LATIN.search(text))):   # docnum: "3 x 12.90"
            continue
        fp, text = pick(stream, text)
        if not (fp and text):
            continue
        if stream in VOWELLED_STREAMS and any(c in MARK_NAMES for c in text):
            img, dmg, vis = vowelled_image(text, fp)
            if img is None:
                drops += 1
                continue
        else:
            img, dmg, vis = None, "none", "n/a"
        break
    else:
        return None
    if img is None:
        nbs = ("", "")
        # no neighbour slivers on vowelled lines: the line above shows its below-marks right over
        # this line, where they read as this line's marks
        if not _V["clean"] and stream not in VOWELLED_STREAMS:
            nbs = tuple(adapt(MAKERS[stream]()[0], fp) for _ in range(2))
            nbs = tuple(n if n and set(n) <= G.ALLOWED and {ord(c) for c in n if not c.isspace()} <= G.cmap(fp) else ""
                        for n in nbs)
        img, size = G.render(text, fp, nbs)
        if not _V["clean"]:
            if stream == "negatives" and random.random() < 0.6:
                img = edge_noise(img)
            is_heavy = random.random() < (0.2 if stream in VOWELLED_STREAMS else 0.4)
            dmg = "heavy" if is_heavy else "light"
            clean_ink = G.ink_frac(img)
            for _ in range(3):                           # never keep a line the damage wiped out
                out_img = heavy(img, size, stream) if is_heavy else light(img)
                if G.ink_frac(out_img) > 0.25 * clean_ink:
                    break
            else:
                out_img, dmg = img, "none"
            img = out_img
    if not _V["clean"] and random.random() < FADED_P:
        f_img = faded(img, keep_marks=any(c in MARK_NAMES for c in text))
        if G.ink_frac(f_img) > 0 or np.asarray(f_img).min() < np.percentile(np.asarray(f_img), 90) - 40:
            img, dmg = f_img, dmg + "+faded"
    base = Path(out) / f"{idx:06d}"
    img.save(f"{base}.png")
    Path(f"{base}.gt.txt").write_text(text + "\n", encoding="utf-8")
    return idx, stream, src, Path(fp).name, dmg, vis, drops


def parse_mix(s):
    mix = {k: float(v) for k, v in (p.split("=") for p in s.split(","))}
    assert set(mix) <= set(STREAMS), mix
    return mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--mix", default=DEFAULT_MIX)
    ap.add_argument("--vowelled", action="append", default=[], help="NAME=PATH[:WEIGHT], one passage per line")
    ap.add_argument("--langdata", required=True, help="heb/eng wordlist and training_text (fetch_inputs.sh)")
    ap.add_argument("--fonts", required=True, help="folder with the font files of fonts_v2.tsv")
    ap.add_argument("--font-list", default=str(FONT_LIST))
    ap.add_argument("--font-caps", default=str(FONT_CAPS))
    ap.add_argument("--only", default="", help="comma-separated line numbers: render only these")
    ap.add_argument("--per-stream", type=int, default=0, help="smoke test: N lines of every stream, in OUT/<stream>/")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    assert features.check("raqm"), "Pillow needs raqm for Hebrew shaping"
    vow = []
    for spec in a.vowelled:
        name, rest = spec.split("=", 1)
        path, _, w = rest.partition(":")
        vow.append((name, path, float(w or 1)))
    cfg = {"langdata": Path(a.langdata), "clean": a.clean, "fonts": a.fonts, "font_list": a.font_list,
           "caps": a.font_caps, "vowelled": vow}
    out = Path(a.out)
    rng = random.Random(a.seed)
    if a.per_stream:
        jobs = []
        for s in STREAMS:
            (out / s).mkdir(parents=True, exist_ok=True)
            jobs += [(i, a.seed * 1000003 + STREAMS.index(s) * 100000 + i, str(out / s), s) for i in range(a.per_stream)]
    else:
        out.mkdir(parents=True, exist_ok=True)
        mix = parse_mix(a.mix)
        names, weights = list(mix), list(mix.values())
        jobs = [(i, a.seed * 1000003 + i, str(out), rng.choices(names, weights)[0]) for i in range(a.start, a.start + a.n)]
        if a.only:
            keep = {int(x) for x in a.only.split(",")}
            jobs = [j for j in jobs if j[0] in keep]
    with Pool(a.jobs, _init, (cfg,)) as pool:
        rows = [r for r in pool.imap_unordered(_one, jobs, chunksize=32) if r]
    rows.sort()
    with open(out / "manifest.tsv", "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    print(f"{len(rows)} of {len(jobs)} lines written to {out}")
    for k, v in sorted(Counter(r[1] for r in rows).items()):
        print(f"  {k:10s} {v}")
    print("damage:", dict(Counter(r[4] for r in rows)), "| fonts used:", len({r[3] for r in rows}))
    v = [r for r in rows if r[5] != "n/a"]
    if v:
        c = Counter(r[5] for r in v)
        drops = sum(r[6] for r in rows)
        print(f"mark check on {len(v)} vowelled lines: heavy kept {sum(1 for r in v if r[4] == 'heavy')}, "
              f"heavy -> light {c['fallback']} ({100 * c['fallback'] / len(v):.1f}%), "
              f"lines dropped (even light lost a mark) {drops} ({100 * drops / (len(v) + drops):.1f}% of attempts)")


if __name__ == "__main__":
    main()
