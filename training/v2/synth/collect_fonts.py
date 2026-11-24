#!/usr/bin/env python3
"""Put the 165 fonts of ../fonts.tsv into one folder, as gen_lines_v2.py --fonts expects.

Takes the files fetch_v2_sources.py --only fonts downloaded (V2_SOURCES/fonts/), unpacks the Culmus
and Ezra SIL files from their archives (V2_SOURCES/fonts/archives/), and copies the system fonts
from their Debian/Ubuntu packages (fonts-dejavu-core, fonts-freefont-ttf, fonts-liberation,
fonts-noto-core, fonts-unifont), and takes the three Noto files whose download was not recorded
from the line dataset (DATASET/fonts/). Every file is checked against fonts.tsv's sha256. A different
version still renders, but the lines drawn in it will not match ours byte for byte.

    python3 -I collect_fonts.py V2_SOURCES DATASET FONTS
"""
import csv
import hashlib
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    src, dataset, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dst.mkdir(parents=True, exist_ok=True)
    with open(HERE.parent / "fonts.tsv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    members = {}
    for arc in (src / "fonts" / "archives").glob("*"):
        if arc.name.endswith(".zip"):
            z = zipfile.ZipFile(arc)
            members.update({Path(n).name: (lambda n=n, z=z: z.read(n)) for n in z.namelist() if not n.endswith("/")})
        else:
            t = tarfile.open(arc)
            members.update({Path(m.name).name: (lambda m=m, t=t: t.extractfile(m).read())
                            for m in t.getmembers() if m.isfile()})
    problems = []
    for r in rows:
        f, out = r["file"], dst / r["file"]
        s = r["source"]
        if s.startswith("dataset:"):
            p = dataset / s.split(":", 1)[1].strip()
            if not p.exists():
                problems.append(f"{f}: not in the dataset ({p})")
                continue
            shutil.copyfile(p, out)
        elif s.startswith("/usr/share/fonts"):
            p = Path(s.split(" (")[0])
            if not p.exists():
                problems.append(f"{f}: not installed ({s.split('package ')[-1].rstrip(')')})")
                continue
            shutil.copyfile(p, out)
        elif ": http" in s:
            if f not in members:
                problems.append(f"{f}: not in the downloaded archives ({s.split(':')[0]})")
                continue
            out.write_bytes(members[f]())
        else:
            p = src / "fonts" / f
            if not p.exists():
                problems.append(f"{f}: not downloaded")
                continue
            shutil.copyfile(p, out)
        if r["sha256"] and sha256(out) != r["sha256"]:
            problems.append(f"{f}: sha256 differs from ours")
    print(f"{len(rows) - sum(': not' in p for p in problems)} of {len(rows)} fonts in {dst}")
    for p in problems:
        print("  -", p)


if __name__ == "__main__":
    main()
