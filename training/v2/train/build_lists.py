#!/usr/bin/env python3
"""heb_rivok 2.0, first stage (iterations 0-60,000): link the real lines into the ground-truth folder
(before prep_train.py), then write the weighted training list (after it).

  link   GT DATASET: the real training pool (DATASET/lines.tsv, set "pool", in file order) as
           r000000.png symlinks + .gt.txt in GT; the map r-id -> book, page, line in DATA/real_links.tsv
  lists  MODEL GT: rewrite DATA/MODEL/list.train and list.eval from the lstmf files in GT:
           synthetic lines once each (their stream shares are set by the generator mix);
           real lines with per-group weights: Hyman series 50%, Seder Tefilot 25%, the other
           books 25% of the real share; repeats are extra symlinks (r..._k1.lstmf).
           real share = 25% of all training lines. Eval list: 300 synthetic + 200 real (only for
           lstmtraining's own progress lines; checkpoints are chosen with dev_score.py).

    python3 -I build_lists.py --data DATA link GT DATASET
    python3 -I build_lists.py --data DATA lists hebv2 GT [--real-share 0.25] [--seed 5]
    python3 -I build_lists.py --data DATA add hebv2 GT 61000 63000 [--seed 12]

2.0 ran the first 8,000 iterations on the "lists" list, made before the 2,000 Latin lines (generator
lines 61000-62999) existed. Then "add" appended their lstmf files (in line order) and shuffled the
whole list with seed 12; the eval list stayed. The lists are deterministic: the published dataset
has the final one (training_lists/heb_rivok_stage1.train.txt) to compare with.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

GROUPS = {"hyman": 0.50, "tefilot": 0.25, "other": 0.25}


def group(book):
    if "Aaron_Hyman" in book:
        return "hyman"
    if "Seder_Tefilot" in book:
        return "tefilot"
    return "other"


def link(data, gt, dataset):
    gt.mkdir(parents=True, exist_ok=True)
    head, *rows = [l.split("\t") for l in (dataset / "lines.tsv").read_text(encoding="utf-8").splitlines()]
    rows = [dict(zip(head, r)) for r in rows if r[head.index("set")] == "pool"]
    with open(data / "real_links.tsv", "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            name = f"r{i:06d}"
            png = gt / f"{name}.png"
            if not png.exists():
                png.symlink_to((dataset / r["image"]).resolve())
            (gt / f"{name}.gt.txt").write_text(r["label"] + "\n", encoding="utf-8")
            fh.write(f"{name}\t{r['book']}\t{r['page']}\t{r['line']}\n")
    print(f"{len(rows)} real lines linked into {gt}")


def lists(data, model, gt, share, seed):
    rng = random.Random(seed)
    lstmf = {p.stem: p for p in gt.glob("*.lstmf") if "_k" not in p.stem}
    syn = sorted(str(p) for s, p in lstmf.items() if not s.startswith("r"))
    book_of = {n: b for n, b, *_ in (l.split("\t") for l in (data / "real_links.tsv").read_text(encoding="utf-8").splitlines())}
    real = defaultdict(list)
    for s, p in lstmf.items():
        if s.startswith("r") and s in book_of:
            real[group(book_of[s])].append(p)
    rng.shuffle(syn)
    ev = syn[:300]
    syn = syn[300:]
    ev_real = []
    for g in GROUPS:          # 2.0's run shuffled the groups in the order its folder listed them: this order
        real[g].sort()
        rng.shuffle(real[g])
    for g, k in (("hyman", 100), ("tefilot", 50), ("other", 50)):
        ev_real += [str(p) for p in real[g][:k]]
        real[g] = real[g][k:]
    n_real = int(round(len(syn) * share / (1 - share)))
    train_real, counts = [], {}
    for g, w in GROUPS.items():
        want, have = int(round(n_real * w)), real[g]
        picked = []
        k = 0
        while len(picked) < want and have:
            for p in have:
                if len(picked) >= want:
                    break
                if k == 0:
                    picked.append(str(p))
                else:
                    alias = p.with_name(f"{p.stem}_k{k}.lstmf")
                    if not alias.exists():
                        alias.symlink_to(p.name)
                    picked.append(str(alias))
            k += 1
        train_real += picked
        counts[g] = {"lines": len(have), "samples": len(picked), "repeats": round(len(picked) / max(1, len(have)), 2)}
    train = syn + train_real
    rng.shuffle(train)
    out = data / model
    out.mkdir(parents=True, exist_ok=True)
    (out / "list.train").write_text("\n".join(train) + "\n")
    (out / "list.eval").write_text("\n".join(ev + ev_real) + "\n")
    stats = {"train": len(train), "synthetic": len(syn), "real_samples": len(train_real),
             "real_share": round(len(train_real) / len(train), 3), "groups": counts, "eval": len(ev) + len(ev_real)}
    (out / "lists.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


def add(data, model, gt, first, last, seed):
    out = data / model
    tr = [l.strip() for l in (out / "list.train").read_text().splitlines() if l.strip()]
    new = [str(p) for p in (gt / f"{i:06d}.lstmf" for i in range(first, last)) if p.exists()]
    (out / f"list.train.before_{first}").write_text("\n".join(tr) + "\n")
    allp = tr + new
    random.Random(seed).shuffle(allp)
    (out / "list.train").write_text("\n".join(allp) + "\n")
    print(f"list.train: {len(tr)} -> {len(allp)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="working folder (as prep_train.py --data-dir)")
    ap.add_argument("what", choices=["link", "lists", "add"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--real-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, help="lists: 5, add: 12 (2.0's runs)")
    a = ap.parse_args()
    data = a.data.resolve()
    data.mkdir(parents=True, exist_ok=True)
    if a.what == "link":
        link(data, Path(a.args[0]).resolve(), Path(a.args[1]).resolve())
    elif a.what == "lists":
        lists(data, a.args[0], Path(a.args[1]).resolve(), a.real_share, 5 if a.seed is None else a.seed)
    else:
        add(data, a.args[0], Path(a.args[1]).resolve(), int(a.args[2]), int(a.args[3]), 12 if a.seed is None else a.seed)


if __name__ == "__main__":
    sys.exit(main())
