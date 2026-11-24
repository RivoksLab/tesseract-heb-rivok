#!/usr/bin/env python3
"""Score one float/int traineddata on the 2.0 dev sets (the dataset's dev/ lines). Choices only.

Tesseract reads each chunk of crops as one list file (--psm 13, one thread per process), so the
model loads once per chunk. Text is compared in logical order after a fold (gershayim/geresh/maqaf,
typographic and low quotes, dashes, NBSP, bidi controls, whitespace). Niqqud is kept: on vowelled
sets it is part of the reading; on unvowelled sets any mark is an invented one.

Per set: micro CER, CER without marks, and invented marks per 1,000 predicted letters on lines
whose label has no marks. Dev score = mean of the three set CERs (each set counts the same); the
"modern" guard set is reported but not in the mean.

The 2.0 runs also scored 253 lines from private receipts and put them in the mean (four sets). Those
lines are not published, so this public scorer leaves them out; training/v2/README.md says what that
changes (the same checkpoints for heb_rivok_lite; for heb_rivok a 4-snapshot average would have
scored 0.02 points better than the released 8-snapshot one).

    python3 -I dev_score.py MODEL.traineddata OUT.json --dataset DATASET [--jobs 2]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SETS = ("milhamot", "tefilot", "tanakh", "modern")
SCORED = ("milhamot", "tefilot", "tanakh")    # the dev score; "modern" is a guard only
DATASET_SETS = {"milhamot": "milhamot", "tefilot": "tefilot", "tanakh_synth": "tanakh", "modern_synth": "modern"}


def dev_rows(dataset):
    """(set, id, image path, label) for the dataset's dev lines, in its file order."""
    head, *rows = [l.split("\t") for l in (dataset / "lines.tsv").read_text(encoding="utf-8").splitlines()]
    out = []
    for r in rows:
        d = dict(zip(head, r))
        if d["split"] == "dev":
            out.append((DATASET_SETS[d["set"]], Path(d["image"]).stem, str(dataset / d["image"]), d["label"]))
    return out
FOLD = str.maketrans({"״": '"', "׳": "'", "־": "-", "“": '"', "”": '"', "„": '"', "″": '"', "‘": "'", "’": "'",
                      "‚": "'", "′": "'", "–": "-", "—": "-", "−": "-", " ": " ", " ": " ", " ": " "})
FOLD[0x05BA] = "ֹ"      # holam haser for vav = holam: the same dot in most fonts (folded from v2.1 on)
BIDI = dict.fromkeys(map(ord, "‎‏؜‪‫‬‭‮⁦⁧⁨⁩‍"))
MARK = re.compile("[֑-ׇ]")
NIQ = re.compile("[ְ-ׇֽֿׁׂ]")
NEW_MARKS = {"ֽ": "meteg", "ֱ": "hataf_segol", "ֳ": "hataf_qamats", "ֻ": "qubuts",
             "ֺ": "holam_haser", "ׇ": "qamats_qatan"}


def norm(t):
    t = unicodedata.normalize("NFC", (t or "").translate(BIDI).translate(FOLD))
    return " ".join(t.split())


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def run_chunk(model, paths):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(paths) + "\n")
        lst = fh.name
    try:
        env = {"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin"}
        p = subprocess.run(["nice", "-n", "10", "tesseract", lst, "stdout", "--psm", "13",
                            "--tessdata-dir", str(model.parent), "-l", model.stem],
                           capture_output=True, env=env, text=True)
        parts = p.stdout.split("\f")
        if len(parts) == len(paths) + 1 and not parts[-1].strip():
            parts = parts[:-1]
        if len(parts) != len(paths):                       # fall back: one process per crop
            parts = []
            for q in paths:
                r = subprocess.run(["tesseract", q, "stdout", "--psm", "13", "--tessdata-dir", str(model.parent),
                                    "-l", model.stem], capture_output=True, env=env, text=True)
                parts.append(r.stdout)
        return [x.strip() for x in parts]
    finally:
        os.unlink(lst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--dataset", type=Path, required=True, help="the published line dataset")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--keep-preds", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated sets to score (e.g. modern)")
    a = ap.parse_args()
    rows = [list(r) for r in dev_rows(a.dataset.resolve())]
    if a.only:
        rows = [r for r in rows if r[0] in a.only.split(",")]
    t0 = time.time()
    chunks = [rows[i::a.jobs * 4] for i in range(a.jobs * 4)]
    with ThreadPoolExecutor(a.jobs) as ex:
        outs = list(ex.map(lambda c: run_chunk(a.model.resolve(), [r[2] for r in c]), chunks))
    preds = {}
    for c, o in zip(chunks, outs):
        for r, p in zip(c, o):
            preds[(r[0], r[1])] = p
    agg = defaultdict(lambda: defaultdict(float))
    for s, i, _, label in rows:
        g, p = norm(label), norm(preds[(s, i)])
        A = agg[s]
        A["n"] += 1
        A["chars"] += len(g)
        A["edits"] += lev(p, g)
        gn, pn = MARK.sub("", g), MARK.sub("", p)
        A["chars_nomarks"] += len(gn)
        A["edits_nomarks"] += lev(pn, gn)
        for c in NEW_MARKS:                                # the 6 marks v2 adds: found / in labels
            if c in g:
                A["new_gold"] += g.count(c)
                A["new_hit"] += min(g.count(c), p.count(c))
        if not NIQ.search(g):
            A["plain_letters"] += sum(1 for ch in p if "א" <= ch <= "ת")
            A["invented_marks"] += len(NIQ.findall(p))
        A["exact"] += p == g
    res = {"model": str(a.model), "seconds": round(time.time() - t0, 1), "sets": {}}
    for s in sorted(agg, key=lambda x: (SETS.index(x) if x in SETS else len(SETS), x)):   # test files: any set name
        A = agg[s]
        res["sets"][s] = {"n": int(A["n"]), "cer": round(100 * A["edits"] / max(1, A["chars"]), 3),
                          "cer_nomarks": round(100 * A["edits_nomarks"] / max(1, A["chars_nomarks"]), 3),
                          "exact": round(100 * A["exact"] / A["n"], 2),
                          "invented_marks_per_1k": round(1000 * A["invented_marks"] / max(1, A["plain_letters"]), 3)}
    ng = sum(agg[s]["new_gold"] for s in agg)
    res["new_marks_found"] = f"{int(sum(agg[s]['new_hit'] for s in agg))}/{int(ng)}"
    cers = [v["cer"] for k, v in res["sets"].items() if k in SCORED]
    res["dev_score"] = round(sum(cers) / len(cers), 3) if cers else None
    inv = sum(agg[s]["invented_marks"] for s in agg)
    pl = sum(agg[s]["plain_letters"] for s in agg)
    res["invented_marks_per_1k"] = round(1000 * inv / max(1, pl), 3)
    if a.keep_preds:
        res["preds"] = {f"{s}|{i}": preds[(s, i)] for s, i, _, _ in rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "preds"}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
