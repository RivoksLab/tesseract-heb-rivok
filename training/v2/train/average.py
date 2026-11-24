#!/usr/bin/env python3
"""Average the weights of a 2.0 run's last K snapshots (float), as the released files were made.

    python3 -I average.py --data DATA --dataset DATASET MODEL [--ks 3,4,5,6,7,8] [--upto ITER]
        averages the last K snapshots for each K, scores every average on dev (dev_score.py) and
        writes DATA/MODEL/run/avg/avgK.traineddata, scores/avgK.json and avg/choice.json; the
        candidate is the K with the lowest dev score (ties: the larger K). Never the last state alone.
    python3 -I average.py --data DATA MODEL --k K --out OUT.traineddata
        just the average of the last K snapshots (no scoring).

2.0 released the average of the last 8 snapshots of heb_rivok's second stage (66,000-80,000) and of
the last 6 of heb_rivok_lite (50,000-60,000), chosen with a dev score that also counted 253 private
receipt lines. Without them (this public dev_score.py) lite's choice is the same; heb_rivok's best K
would be 4 (dev 4.386 against 4.404 for K = 8). See training/v2/README.md.

Reading and writing the LSTM inside a .traineddata needs tesslstm.py (next to this file): every
float weight matrix is averaged; every other component and byte is taken from the first snapshot.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tesslstm import LSTM_INDEX, Traineddata  # noqa: E402


def average(paths, out):
    tds = [Traineddata.read(str(p)) for p in paths]
    avg = tds[0]
    m = avg.lstm
    ts = m.tensors()
    others = [t.lstm.tensors() for t in tds[1:]]
    for o, t in zip(others, tds[1:]):
        assert o.keys() == ts.keys()
        assert all(o[k].shape == ts[k].shape and o[k].dtype == ts[k].dtype for k in ts)
        assert t.components.keys() == avg.components.keys()
        assert all(t.components[i] == avg.components[i] for i in t.components if i != LSTM_INDEX)
    for k, a in ts.items():
        a[...] = (a + sum(o[k] for o in others)) / len(tds)
    avg.with_lstm(m).write(str(out))


def snapshots(run, model, upto):
    flts = sorted((run / "float").glob(f"{model}_*.traineddata"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return [p for p in flts if not upto or int(p.stem.rsplit("_", 1)[1]) <= upto]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, help="the published line dataset (for scoring)")
    ap.add_argument("model")
    ap.add_argument("--ks", default="3,4,5,6,7,8")
    ap.add_argument("--k", type=int, help="average only the last K snapshots into --out")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--upto", type=int, default=0, help="last snapshot of the window (default: the last one)")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    run = a.data.resolve() / a.model / "run"
    flts = snapshots(run, a.model, a.upto)
    if a.k:
        average(flts[-a.k:], a.out)
        print("averaged", [p.name for p in flts[-a.k:]], "->", a.out)
        return 0
    (run / "avg").mkdir(exist_ok=True)
    res = {}
    for k in map(int, a.ks.split(",")):
        if k > len(flts):
            continue
        win = flts[-k:]
        out = run / "avg" / f"avg{k}.traineddata"
        average(win, out)
        sc = run / "scores" / f"avg{k}.json"
        subprocess.run([sys.executable, "-I", str(HERE / "dev_score.py"), str(out), str(sc), "--jobs", str(a.jobs),
                        "--dataset", str(a.dataset)], check=True, capture_output=True)
        r = json.loads(sc.read_text(encoding="utf-8"))
        res[k] = {"window": [int(p.stem.rsplit("_", 1)[1]) for p in win], "dev_score": r["dev_score"],
                  "sets": {s: v["cer"] for s, v in r["sets"].items()}, "invented_marks_per_1k": r["invented_marks_per_1k"]}
        print(k, json.dumps(res[k]), flush=True)
    best = min(res, key=lambda k: (res[k]["dev_score"], -k))
    choice = {"k": best, **res[best], "all": res}
    (run / "avg" / "choice.json").write_text(json.dumps(choice, indent=1))
    print("CHOICE", best, res[best]["dev_score"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
