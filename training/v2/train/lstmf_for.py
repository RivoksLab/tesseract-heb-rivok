#!/usr/bin/env python3
"""Make the .box and .lstmf files for generator lines FIRST..LAST-1 of a ground-truth folder (lines
added after prep_train.py ran: 2.0's Latin and rare streams), with prep_train.py's own recipe.

    python3 -I lstmf_for.py GT FIRST LAST [--jobs 4]
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # training/prep_train.py
from prep_train import box_and_lstmf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gt", type=Path)
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    pngs = [p for p in (a.gt.resolve() / f"{i:06d}.png" for i in range(a.first, a.last)) if p.exists()]
    with ThreadPoolExecutor(a.jobs) as ex:
        done = [x for x in ex.map(box_and_lstmf, pngs) if x]
    print(f"{len(done)} of {len(pngs)} lstmf files made")


if __name__ == "__main__":
    sys.exit(main())
