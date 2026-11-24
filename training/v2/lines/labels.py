"""Training-label rules of heb_rivok 2.0's real lines, copied verbatim from the internal
finalize.py (rules for the training books) and real_lines.py (the checks before training).
build_lines.py applies them in this order: label_problem, unvowelled Seder Tefilot lines, dense
editorial niqqud, normalise, cantillation; then sof pasuq and ZWJ, validated pages, charset, plene,
abbreviation, editor_note. See training/v2/label_changes.tsv for what they changed.
"""
import re
import unicodedata


# ---------------------------------------------------------------- from finalize.py
VOWELLED_ONLY = {"1885_Seder_Tefilot_K_Minhag_K_K_Sefaradim_Yos_3eabf015"}
NIQ = re.compile(r"[ְ-ׇּׁׂ]")
COMB = re.compile(r"[֑-ׇֽֿׁׂׅׄ]")  # combining marks (not maqaf/paseq/sof pasuq)
SHIN, HOLAM, SHIN_DOT, SIN_DOT, DAGESH, VAV = "ש", "ֹ", "ׁ", "ׂ", "ּ", "ו"
VOWELS = set("ְֱֲֳִֵֶַָׇֻ")
CANT = re.compile(r"[\u0591-\u05AF]")  # cantillation marks (te'amim)

def fix_sin_dot(label: str):
    """ש + holam (U+05B9) without a shin/sin dot: the holam is a mistyped sin dot when the ש also carries
    another vowel, or the next letter is a vav with holam or dagesh (the vowel sits on the vav), or the
    word ends there. Otherwise (holam only, consonant follows) it is ambiguous and left alone.
    Returns (label, fixed, ambiguous)."""
    ch = list(label)
    n, i, res, fixed, amb = len(ch), 0, [], 0, 0
    while i < n:
        if ch[i] != SHIN:
            res.append(ch[i]); i += 1
            continue
        j = i + 1
        while j < n and COMB.match(ch[j]):
            j += 1
        marks = ch[i + 1:j]
        if HOLAM in marks and SHIN_DOT not in marks and SIN_DOT not in marks:
            nxt = ch[j] if j < n else ""
            k, nm = j + 1, []
            while k < n and COMB.match(ch[k]):
                nm.append(ch[k]); k += 1
            if (any(m in VOWELS for m in marks) or (nxt == VAV and (HOLAM in nm or DAGESH in nm))
                    or not ("א" <= nxt <= "ת")):
                marks = [SIN_DOT if m == HOLAM else m for m in marks]
                fixed += 1
            else:
                amb += 1
        res.append(SHIN); res += marks
        i = j
    return unicodedata.normalize("NFC", "".join(res)), fixed, amb

HEB_L = "א-ת"
ABBR_TOKEN = re.compile(rf"[{HEB_L}]+[\"״][{HEB_L}]+")
NOTE_MARK = re.compile(rf"\([{HEB_L}0-9]{{1,3}}\)")

def label_problem(label: str, ocr: str):
    """Line-level label problems found in the 2026-11-09 check (see label_checks.py), or None:
    printed vowels missing (the OCR sees dense niqqud, the label has none); an abbreviation expanded
    (an OCR abbreviation whose letters are the initials of consecutive label words, and the label does
    not have the abbreviation); printed note markers "(א)" missing from the label."""
    letters = len(re.findall(f"[{HEB_L}]", ocr)) or 1
    nv = len(NIQ.findall(ocr))
    if nv >= 5 and nv / letters >= 0.3 and not NIQ.search(label):
        return "printed_vowels_missing"
    lw = label.split()
    lab_abbr = {re.sub(f"[^{HEB_L}]", "", t) for t in ABBR_TOKEN.findall(label)}
    for tok in ABBR_TOKEN.findall(ocr):
        lets = re.sub(f"[^{HEB_L}]", "", tok)
        n = len(lets)
        if n < 2 or lets in lab_abbr:
            continue
        for k in range(len(lw) - n + 1):
            if all(w[0] == lets[i] and not re.search("[\"״']", w) for i, w in enumerate(lw[k:k + n])):
                return "abbreviation_expanded"
    if len(NOTE_MARK.findall(ocr)) > len(NOTE_MARK.findall(label)):
        return "note_markers_missing"
    return None


def normalise(label: str, c):
    label = label.replace("͏", "")
    if re.search("[יִ-ﭏ]", label):  # Hebrew presentation forms (alternative ayin U+FB20, ...) -> base letters
        label = re.sub("[יִ-ﭏ]", lambda m: unicodedata.normalize("NFKC", m.group(0)), label)
        c["presentation_forms_mapped"] += 1
    if "|" in label:
        label = re.sub(r"\s*\|\s*", " ׀ ", label); c["paseq_mapped"] += 1
    label, f, a = fix_sin_dot(label)
    c["sin_dot_fixed"] += f; c["sin_dot_ambiguous"] += a
    return re.sub(r"\s+", " ", label).strip()


# ---------------------------------------------------------------- from real_lines.py
NEW_MARKS = "ֱֳׇֺֻֽ"   # hataf segol, hataf qamats, qamats qatan, holam haser, qubuts, meteg
TEFILOT = "1885_Seder_Tefilot_K_Minhag_K_K_Sefaradim_Yos_3eabf015"
HEB = "א-ת"
MARKS = re.compile("[֑-ׇ]")
ABBR = re.compile(rf"[{HEB}]+[\"״][{HEB}]|[{HEB}]+['׳](?=\s|$|[.,:;)])")
NOTE = re.compile(r"צ[\"״]ל|ט[\"״]ס|\(!\)|sic|כך במקור|ויקיטקסט|הערת העורך")

def allowed_chars(UNICHARSET):
    chars = set(NEW_MARKS) | {" "}
    lines = UNICHARSET.read_text(encoding="utf-8").splitlines()
    for ln in lines[1:]:
        tok = ln.split(" ", 1)[0]
        if tok not in ("NULL", "Joined", "|Broken|0|1"):
            chars.update(tok)
    return chars


def words(s):
    return [MARKS.sub("", t).strip(".,:;!?()[]\"'״׳–—-") for t in s.split()]


def plene(label, ocr):
    if len(re.findall("[\u05b0-\u05bc\u05c1\u05c2\u05c7]", label)) >= 3:
        return None
    ow = set(words(ocr))
    for t in words(label):
        if len(t) < 4 or t in ow:
            continue
        for i in range(1, len(t) - 1):
            if t[i] in "יו" and t[:i] + t[i + 1:] in ow:
                return t
    return None
