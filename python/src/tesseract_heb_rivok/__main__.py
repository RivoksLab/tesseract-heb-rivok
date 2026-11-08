"""python -m tesseract_heb_rivok [--merged [DIR]] [--check]

Prints the folder to use as Tesseract's --tessdata-dir (or TESSDATA_PREFIX):
  (no option)     this package's folder: heb_rivok and heb_rivok_lite only
  --merged [DIR]  a folder with these models plus the system's tessdata (eng, osd, configs),
                  created or refreshed in DIR or a per-user cache folder
  --check         verify the model files' SHA-256 and that Tesseract can list them
"""

import argparse
import subprocess
import sys

from . import MODELS, __version__, merged_tessdata_dir, tessdata_dir, verify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tesseract_heb_rivok",
                                 description="Print the tessdata folder for heb_rivok / heb_rivok_lite.")
    ap.add_argument("--merged", nargs="?", const="", metavar="DIR",
                    help="print a merged folder (these models + the system's tessdata)")
    ap.add_argument("--check", action="store_true", help="verify the files and that Tesseract loads them")
    ap.add_argument("--version", action="version",
                    version=f"tesseract-heb-rivok {__version__} ("
                            + ", ".join(f"{n} {v}" for n, (v, _) in MODELS.items()) + ")")
    a = ap.parse_args(argv)
    if a.check:
        bad = [n for n, ok in verify().items() if not ok]
        if bad:
            print("SHA-256 mismatch: " + ", ".join(bad), file=sys.stderr)
            return 1
        try:
            out = subprocess.run(["tesseract", "--tessdata-dir", tessdata_dir(), "--list-langs"],
                                 capture_output=True, text=True, timeout=30)
        except OSError:
            print("files OK; tesseract not found on PATH", file=sys.stderr)
            return 1
        listed = set(out.stdout.split())
        missing = [n for n in MODELS if n not in listed]
        print("OK: " + ", ".join(MODELS) if not missing else "Tesseract does not list: " + ", ".join(missing))
        return 0 if not missing else 1
    if a.merged is not None:
        print(merged_tessdata_dir(a.merged or None))
    else:
        print(tessdata_dir())
    return 0


if __name__ == "__main__":
    sys.exit(main())
