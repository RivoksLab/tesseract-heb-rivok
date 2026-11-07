"""Scoring used for the published heb_rivok numbers.

GlotOCR-bench: their metric, copied from cisnlp/GlotOCR-bench evaluation/metrics/utils.py
(whitespace-free Levenshtein, minimum over the reference, its reverse, lower case and accents
stripped, capped at 1), averaged over lines. strip_marks() is our extra "no marks" view.

HaZefira lines: character error rate as summed edits / summed reference characters, after
norm() (niqqud, cantillation and BiDi controls removed, NFC, single spaces). Both sides are put
in display order with python-bidi (base direction L and R) and the smaller distance counts, so
a reading that is right but in another order is not charged.

HaZefira pages: order-free word matching. Both texts are cut into words (markup and marks
removed, ׳ ״ ־ – — folded to ' " -, punctuation stripped from word edges, Hebrew/digit runs
split); recall, precision and F1 are over the word multisets.
"""
import html
import re
import unicodedata

from bidi.algorithm import get_display


# ------------------------------------------------------------------ GlotOCR-bench (their metric)
def normalize_text(text):
    return "" if text is None else text.strip()


def remove_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", text)


def levenshtein(a, b):
    a = re.sub(r"\s+", "", a); b = re.sub(r"\s+", "", b)
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a; n, m = m, n
    cur = list(range(n + 1))
    for i in range(1, m + 1):
        prev, cur = cur, [i] + [0] * n
        for j in range(1, n + 1):
            add, dele, chg = prev[j] + 1, cur[j - 1] + 1, prev[j - 1]
            if a[j - 1] != b[i - 1]:
                chg += 1
            cur[j] = min(add, dele, chg)
    return cur[n]


def glot_cer(pred, gt):
    gt_len = max(len(gt), 1)
    cands = [(pred, gt), (pred, gt[::-1]),
             (pred.lower(), gt.lower()), (pred.lower(), gt[::-1].lower()),
             (remove_accents(pred), gt), (remove_accents(pred.lower()), gt.lower())]
    return min(min(levenshtein(a, b) for a, b in cands) / gt_len, 1.0)


def strip_marks(t):
    """Drop combining marks (niqqud / cantillation)."""
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


# ------------------------------------------------------------------ line CER (HaZefira lines)
CTRL = set("\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u061c")
_PREAMBLE = re.compile(r"^\s*[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]{2,30}[\uff1a:\u3002]\s*")  # VLM chat preamble
FOLD = str.maketrans({"׳": "'", "״": '"', "־": "-", "–": "-"})


def norm(t):
    t = _PREAMBLE.sub("", t or "")
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if not ("\u0591" <= c <= "\u05c7" and unicodedata.category(c) == "Mn") and c not in CTRL)
    return " ".join(unicodedata.normalize("NFC", t).split())


def lev(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _disp(t, d):
    try:
        return get_display(t, base_dir=d)
    except AssertionError:      # python-bidi 0.6 trips on some exotic characters
        return t


def dist_display(p, g):
    return min(lev(_disp(p, d), _disp(g, d)) for d in ("L", "R"))


# ------------------------------------------------------------------ page words (HaZefira pages)
PAGE_CTRL = dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff"))
PAGE_FOLD = str.maketrans({"׳": "'", "״": '"', "־": "-", "–": "-", "—": "-"})


def tokens(text):
    t = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))           # HTML tables
    t = re.sub(r"[|#*`]+|\$+", " ", t)                                 # markdown
    t = unicodedata.normalize("NFD", t.translate(PAGE_CTRL))
    t = "".join(c for c in t if not unicodedata.category(c) == "Mn")
    t = unicodedata.normalize("NFC", t).translate(PAGE_FOLD)
    t = re.sub(r"(?<=[א-ת])(?=\d)|(?<=\d)(?=[א-ת])", " ", t)         # טלפון0775 -> טלפון 0775
    out = []
    for w in t.split():
        w = w.strip(".,:;!?()[]{}\"'-–_/\\₪%*•")
        if w:
            out.append(w)
    return out
