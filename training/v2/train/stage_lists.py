#!/usr/bin/env python3
"""Training list for heb_rivok 2.0's second stage (iterations 60,000-80,000, continuing the first stage)
and for heb_rivok_lite 2.0 (one stage).

One list of SIZE samples, each line at most once except where a real group is smaller than its quota:
  real     REAL_SHARE (Hyman 50% / Seder Tefilot 25% / other books 25%, as build_lists.py)
  hq       HQ_SHARE   the generator's rare-stream lines for hataf qamats (manifest substream
                      "rare:hataf qamats"; the main run never saw the rare stream)
  latin    LATIN_SHARE lines without Hebrew (stream latin), in from the first iteration
  rest     the main run's synthetic lines (all other streams), sampled uniformly, so the generator
           mix (replay, vowelled, confusion, siddur, fonts, negatives, signs, docnum) keeps its proportions
The real eval lines and the synthetic eval lines of DATA/hebv2/list.eval stay out; the eval list is reused.

Holam haser for vav (U+05BA) is folded to holam (U+05B9): a picked line that carries it gets a folded
copy in DATA/MODEL-ground-truth (png symlink, folded .gt.txt, lstmf rebuilt with prep_train's recipe).
The rare-stream holam haser lines are not used.

With SIZE = 20000 and a checkpoint at 60000, lstmtraining's sample position (60000 mod 20000 = 0)
starts at the head of the list, so 20k iterations see every sample once.

    python3 -I stage_lists.py --data DATA --gt GT hebv21 [--size 20000] [--real-share 0.35] [--hq-share 0.03]
        [--latin-share 0.025] [--seed 21]
DATA is the working folder of prep_train.py and build_lists.py (real_links.tsv, hebv2/all-gt.txt,
hebv2/list.eval); GT the ground-truth folder (generator lines, real lines, manifest.tsv).

Lite (Hebrew and digits only, 2026-11-13): --charset UNICHARSET drops every candidate line with a
character outside that unicharset (after the holam fold), so no Latin line reaches the list; the hataf
qamats lines then repeat to fill their share (--hq-repeat).
    python3 -I stage_lists.py --data DATA --gt GT heblv2 --size 60000 --latin-share 0 --hq-repeat \
        --charset DATA/heblv2/unicharset
"""
import argparse
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # training/prep_train.py
from prep_train import box_and_lstmf          # noqa: E402  (same box + lstmf recipe as every round)
GROUPS = {"hyman": 0.50, "tefilot": 0.25, "other": 0.25}
HOLAM_HASER, HOLAM = "ֺ", "ֹ"


def group(book):
    if "Aaron_Hyman" in book:
        return "hyman"
    if "Seder_Tefilot" in book:
        return "tefilot"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("model")
    ap.add_argument("--size", type=int, default=20000)
    ap.add_argument("--real-share", type=float, default=0.35)
    ap.add_argument("--hq-share", type=float, default=0.03)
    ap.add_argument("--latin-share", type=float, default=0.025)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--charset", default="", help="keep only lines whose characters are all in this unicharset")
    ap.add_argument("--hq-repeat", action="store_true", help="repeat hataf qamats lines to fill their share")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    DATA, GT = a.data.resolve(), a.gt.resolve()
    out = DATA / a.model
    fold_dir = DATA / f"{a.model}-ground-truth"
    out.mkdir(parents=True, exist_ok=True)
    fold_dir.mkdir(parents=True, exist_ok=True)

    ev = [l.strip() for l in (DATA / "hebv2/list.eval").read_text().splitlines() if l.strip()]
    ev_stems = {Path(p).stem for p in ev}

    # synthetic pools from the generator manifest (index -> stream, substream)
    pools = defaultdict(list)
    for row in (GT / "manifest.tsv").read_text(encoding="utf-8").splitlines():
        f = row.split("\t")
        stem = f"{int(f[0]):06d}"
        if stem in ev_stems or not (GT / f"{stem}.lstmf").exists():
            continue
        s = f[1]
        if s == "rare":
            if f[2] == "rare:hataf qamats":
                pools["hq"].append(stem)
            continue
        pools["latin" if s == "latin" else "rest"].append(stem)

    # real pools (no eval lines, no _k aliases)
    book_of = {n: b for n, b, *_ in (l.split("\t") for l in (DATA / "real_links.tsv").read_text(encoding="utf-8").splitlines())}
    real = defaultdict(list)
    for n, b in book_of.items():
        if n not in ev_stems and (GT / f"{n}.lstmf").exists():
            real[group(b)].append(n)

    # labels: the prep dump (DATA/hebv2/all-gt.txt, in prep_train's sorted-png order) instead of 85k
    # small reads on the HDD; lines generated after prep (latin, rare) are read from their files
    dump = (DATA / "hebv2/all-gt.txt").read_text(encoding="utf-8").split("\n")
    n_syn = sum(1 for _ in (GT / "manifest.tsv").read_text(encoding="utf-8").splitlines()[:61000])
    labels = {(f"{i:06d}" if i < n_syn else f"r{i - n_syn:06d}"): x.strip() for i, x in enumerate(dump) if x.strip()}

    def label(stem):
        if stem not in labels:
            labels[stem] = (GT / f"{stem}.gt.txt").read_text(encoding="utf-8").strip()
        return labels[stem]

    if a.charset:
        allowed = {l.split(" ")[0] for l in Path(a.charset).read_text(encoding="utf-8").splitlines()[1:] if l}
        allowed |= {" ", "\u05ba"}                          # space (unicharset's "NULL"), holam haser (folded below)

        def ok(stem):
            return all(c in allowed for c in unicodedata.normalize("NFC", label(stem)))
        before = {k: len(v) for k, v in pools.items()}
        pools = {k: [s for s in v if ok(s)] for k, v in pools.items()}
        real = {g: [s for s in v if ok(s)] for g, v in real.items()}
        print("charset filter:", {k: f"{before[k]} -> {len(v)}" for k, v in pools.items()}, flush=True)

    n_real = round(a.size * a.real_share)
    n_hq = round(a.size * a.hq_share) if a.hq_repeat else min(round(a.size * a.hq_share), len(pools["hq"]))
    n_lat = min(round(a.size * a.latin_share), len(pools["latin"]))
    n_rest = a.size - n_real - n_hq - n_lat

    picked = []                                    # (stem, kind)
    for k, n in (("hq", n_hq), ("latin", n_lat), ("rest", n_rest)):
        pool = sorted(pools[k])
        rng.shuffle(pool)
        if n and not pool:
            continue
        picked += [(s, k) for s in (pool[:n] if n <= len(pool) else [pool[i % len(pool)] for i in range(n)])]
    counts = {}
    for g, w in GROUPS.items():
        want, have = round(n_real * w), sorted(real[g])
        rng.shuffle(have)
        got = [have[i % len(have)] for i in range(want)]          # repeats only if the group is small
        picked += [(s, "real") for s in got]
        counts[g] = {"lines": len(have), "samples": want, "repeats": round(want / len(have), 2)}

    # holam haser -> holam: folded copies for the picked lines that carry it
    texts = {s: label(s) for s, _ in picked}
    to_fold = sorted({s for s, _ in picked if HOLAM_HASER in texts[s]})
    for s in to_fold:
        png = fold_dir / f"{s}.png"
        if not png.exists():
            png.symlink_to((GT / f"{s}.png").resolve())
        t = unicodedata.normalize("NFC", texts[s].replace(HOLAM_HASER, HOLAM))
        (fold_dir / f"{s}.gt.txt").write_text(t + "\n", encoding="utf-8")
    with ThreadPoolExecutor(a.jobs) as ex:
        built = dict(zip(to_fold, ex.map(box_and_lstmf, [fold_dir / f"{s}.png" for s in to_fold])))
    failed = [s for s, p in built.items() if not p]

    train, seen = [], Counter()
    for s, k in picked:
        if s in failed:
            continue
        base = fold_dir if s in built else GT
        seen[s] += 1
        if seen[s] == 1:
            train.append(str(base / f"{s}.lstmf"))
        else:                                       # a repeat of a small real group: alias symlink
            alias = base / f"{s}_k{seen[s] - 1}.lstmf"
            if not alias.exists():
                alias.symlink_to(f"{s}.lstmf")
            train.append(str(alias))
    rng.shuffle(train)
    (out / "list.train").write_text("\n".join(train) + "\n")
    (out / "list.eval").write_text("\n".join(ev) + "\n")
    kinds = Counter(k for s, k in picked if s not in failed)
    hq_lines = sum(1 for s, _ in picked if "ֳ" in texts[s])
    stats = {"train": len(train), "kinds": dict(kinds), "shares": {k: round(v / len(train), 3) for k, v in kinds.items()},
             "real_groups": counts, "pools": {k: len(v) for k, v in pools.items()},
             "lines_with_hataf_qamats": hq_lines, "holam_haser_folded": len(to_fold), "fold_failed": failed,
             "eval": len(ev), "seed": a.seed}
    (out / "lists.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    sys.exit(main())
