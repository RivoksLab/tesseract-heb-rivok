#!/usr/bin/env python3
"""Drive heb_rivok 2.0 training in chunks, snapshot every chunk, score every snapshot on dev.

lstmtraining keeps no snapshots, so it runs to --max_iterations = current + CHUNK, the checkpoint
is copied, converted to a float traineddata and scored (dev_score.py, in the background while the
next chunk trains). The checkpoint carries the trainer state and the sample position, so chunks
continue the same run.

Stop rule: stop when the best dev score is >= PATIENCE iterations old (an improvement
must beat the best by MIN_DELTA), or at --max. The release is an average of the last snapshots,
chosen later (average.py), never the last state alone.

Control files in DATA/MODEL/run/: PAUSE (wait before the next chunk), STOP (finish after the
current chunk). Events go to DATA/MODEL/run/driver.log, one line each:
  CHUNK <iter> <minutes> | SNAPSHOT <iter> | SCORE <iter> dev=... | BEST ... | STOP ... | ERROR ... | DONE

    python3 -I train_v2.py --data DATA --dataset DATASET MODEL --max 60000 [--chunk 2000] [--patience 10000]
        [--min-delta 0.05] --start-lstm START.lstm --old START.traineddata
DATA is the working folder (DATA/MODEL/MODEL.traineddata, list.train, list.eval from prep_train.py,
build_lists.py or stage_lists.py); DATASET the published line dataset (its dev sets, for dev_score.py).
lstmtraining's output goes to DATA/logs/MODEL.train.log.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Driver:
    def __init__(self, a):
        self.a = a
        self.m = a.model
        self.data = a.data / self.m
        self.run = self.data / "run"
        for d in ("snapshots", "float", "scores"):
            (self.run / d).mkdir(parents=True, exist_ok=True)
        (self.data / "checkpoints").mkdir(exist_ok=True)
        self.ckpt = self.data / "checkpoints" / f"{self.m}_checkpoint"
        self.log = open(self.run / "driver.log", "a", buffering=1, encoding="utf-8")
        self.scoring = []
        self.lock = threading.Lock()

    def say(self, msg):
        self.log.write(f"{time.strftime('%F %T')} {msg}\n")

    def done_iters(self):
        its = [int(p.stem.rsplit("_", 1)[1]) for p in (self.run / "snapshots").glob("*.checkpoint")]
        return max(its, default=self.a.offset)

    def scores(self):
        out = {}
        for p in (self.run / "scores").glob("*.json"):
            try:
                out[int(p.stem)] = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                pass
        return out

    def train_chunk(self, target):
        cmd = ["lstmtraining", "--debug_interval", "0", "--traineddata", str(self.data / f"{self.m}.traineddata"),
               "--learning_rate", "0.0001", "--model_output", str(self.data / "checkpoints" / self.m),
               "--train_listfile", str(self.data / "list.train"), "--eval_listfile", str(self.data / "list.eval"),
               "--max_iterations", str(target), "--target_error_rate", "0.0001"]
        if self.ckpt.exists():
            cmd += ["--continue_from", str(self.ckpt)]
        else:
            cmd += ["--continue_from", str(self.a.start_lstm), "--old_traineddata", str(self.a.old)]
        env = dict(os.environ, OMP_THREAD_LIMIT=str(self.a.threads), OMP_WAIT_POLICY="PASSIVE")
        t0 = time.time()
        (self.a.data / "logs").mkdir(exist_ok=True)
        with open(self.a.data / "logs" / f"{self.m}.train.log", "a", encoding="utf-8") as fh:
            fh.write(f"\n=== chunk to {target} {time.strftime('%F %T')}\n")
            fh.flush()
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=self.a.data)
        tail = (self.a.data / "logs" / f"{self.m}.train.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        ok = "Finished!" in tail.split(f"=== chunk to {target}")[-1] and self.ckpt.exists()
        return ok, (time.time() - t0) / 60, p.returncode

    def snapshot(self, it):
        snap = self.run / "snapshots" / f"{self.m}_{it}.checkpoint"
        shutil.copy2(self.ckpt, snap)
        flt = self.run / "float" / f"{self.m}_{it}.traineddata"
        p = subprocess.run(["lstmtraining", "--stop_training", "--continue_from", str(snap),
                            "--traineddata", str(self.data / f"{self.m}.traineddata"), "--model_output", str(flt)],
                           capture_output=True, text=True, cwd=self.a.data)
        if p.returncode or not flt.exists():
            self.say(f"ERROR convert {it}: {p.stderr[-300:]}")
            return None
        self.say(f"SNAPSHOT {it}")
        return flt

    def score_async(self, it, flt):
        def work():
            out = self.run / "scores" / f"{it}.json"
            p = subprocess.run([sys.executable, "-I", str(HERE / "dev_score.py"), str(flt), str(out), "--jobs",
                                str(self.a.score_jobs), "--dataset", str(self.a.dataset)], capture_output=True, text=True,
                               cwd=self.a.data)
            if p.returncode or not out.exists():
                self.say(f"ERROR score {it}: {p.stderr[-300:]}")
                return
            r = json.loads(out.read_text(encoding="utf-8"))
            s = r["sets"]
            self.say(f"SCORE {it} dev={r['dev_score']:.3f} " + " ".join(f"{k}={v['cer']:.2f}" for k, v in s.items())
                     + f" invented/1k={r['invented_marks_per_1k']:.2f} ({r['seconds']:.0f}s)")
        t = threading.Thread(target=work)
        t.start()
        self.scoring.append(t)

    def should_stop(self):
        sc = self.scores()
        if len(sc) < 2:
            return None
        its = sorted(sc)
        best_it, best = None, None
        for it in its:                                    # an improvement must beat the best by MIN_DELTA
            d = sc[it]["dev_score"]
            if best is None or d < best - self.a.min_delta:
                best_it, best = it, d
        last = its[-1]
        self.say(f"BEST {best_it} dev={best:.3f} (latest scored {last})")
        if last - best_it >= self.a.patience:
            return f"no dev gain > {self.a.min_delta} since {best_it} ({last - best_it} iterations)"
        return None

    def main(self):
        self.say(f"START model={self.m} max={self.a.max} chunk={self.a.chunk} patience={self.a.patience}")
        while True:
            cur = self.done_iters()
            if cur >= self.a.max:
                self.say(f"STOP reached max {self.a.max}")
                break
            if (self.run / "STOP").exists():
                self.say("STOP requested (STOP file)")
                break
            while (self.run / "PAUSE").exists():
                time.sleep(30)
            if not self.a.no_stop:
                with self.lock:
                    why = self.should_stop()
                if why:
                    self.say(f"STOP {why}")
                    break
            target = cur + self.a.chunk
            ok, mins, rc = self.train_chunk(target)
            if not ok:
                self.say(f"ERROR chunk to {target} failed (rc {rc}); see DATA/logs/{self.m}.train.log")
                return 1
            self.say(f"CHUNK {target} {mins:.1f} min")
            flt = self.snapshot(target)
            if flt:
                self.score_async(target, flt)
        for t in self.scoring:
            t.join()
        if not self.a.no_stop:
            self.should_stop()
        self.say("DONE")
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("model")
    ap.add_argument("--max", type=int, default=60000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=10000)
    ap.add_argument("--min-delta", type=float, default=0.05)
    ap.add_argument("--no-stop", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--score-jobs", type=int, default=2)
    ap.add_argument("--start-lstm", type=Path, help="the start network (combine_tessdata -e START.traineddata x.lstm)")
    ap.add_argument("--old", type=Path, help="the start traineddata")
    ap.add_argument("--offset", type=int, default=0,
                    help="a follow-up stage: DATA/MODEL/checkpoints/MODEL_checkpoint was copied from another run at "
                         "this iteration; chunks count from here and --max is absolute")
    a = ap.parse_args()
    a.data, a.dataset = a.data.resolve(), a.dataset.resolve()
    return Driver(a).main()


if __name__ == "__main__":
    sys.exit(main())
