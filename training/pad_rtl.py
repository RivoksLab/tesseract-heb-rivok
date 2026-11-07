#!/usr/bin/env python3
"""Make a Tesseract model with Latin letters lay out its pages right to left (v1.0.1 candidate).

Tesseract decides a model's page direction once, from its unicharset: UNICHARSET::major_right_to_left()
counts unichars with bidi class L against R/AL/AN, and ties go to left-to-right. heb_rivok (like Google's
script/Hebrew) has 52 Latin letters + Joined + Broken (54 L) against 33 Hebrew characters, so it is laid
out LTR and multi-column pages are read left column first. At load time the NULL/space entry also
defaults to L, and direction() below counts it.

This appends unused R-class Hebrew characters (presentation forms etc.) to the lstm-unicharset
component only. The network and the recoder are untouched: the recoder has no code for the new
entries, so they can never be output, and the network's null code is stored in the lstm component
(not derived from the unicharset size). Checked on Tesseract 5.5.0 (2026-11-07): byte-identical text
on 662 real crops at psm 7 and 13 (int8 and float), GlotOCR 100/100, 7 example pages; HaZefira pages
word error rate in reading order 88.5% -> 32.3%; the padded characters never appear in any output.

    python3 pad_rtl.py IN.traineddata OUT.traineddata      # needs combine_tessdata on PATH
"""
import shutil, subprocess, sys, tempfile, unicodedata
from pathlib import Path

CANDIDATES = [0x05EF, 0x05C0, 0x05C3, 0x05C6] + list(range(0xFB1D, 0xFB50))   # unused in modern text


def direction(line):
    f = line.split("\t")[0].split(" ")
    return int(f[5]) if len(f) > 5 else 0          # short lines (the NULL entry) default to L


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(src, td / "m.traineddata")
        subprocess.run(["combine_tessdata", "-u", "m.traineddata", "m."], cwd=td, check=True, capture_output=True)
        lines = (td / "m.lstm-unicharset").read_text(encoding="utf-8").split("\n")
        n = int(lines[0]); body = [l for l in lines[1:] if l.strip()]
        assert len(body) == n
        have = {l.split(" ")[0] for l in body}
        L = sum(direction(l) == 0 for l in body); R = sum(direction(l) in (1, 5, 13) for l in body)
        need = max(0, L - R + 1)
        if need == 0:                             # already right-to-left: copy unchanged
            if src.resolve() != dst.resolve():
                shutil.copy(src, dst)
            print(f"{src.name}: L={L} R={R}; already right-to-left, unchanged")
            return
        add = [chr(c) for c in CANDIDATES if chr(c) not in have and unicodedata.bidirectional(chr(c)) == "R"][:need]
        assert len(add) == need, "not enough unused RTL characters"
        for i, ch in enumerate(add):
            uid, props = n + i, 1 if ch.isalpha() else 16
            body.append(f"{ch} {props} 0,255,0,255,0,0,0,0,0,0 Hebrew {uid} 1 {uid} {ch}\t# {ch} [{ord(ch):x} ]{'x' if props == 1 else 'p'}")
        (td / "m.lstm-unicharset").write_text(f"{len(body)}\n" + "\n".join(body) + "\n", encoding="utf-8")
        subprocess.run(["combine_tessdata", "-o", "m.traineddata", "m.lstm-unicharset"], cwd=td, check=True, capture_output=True)
        shutil.copy(td / "m.traineddata", dst)
    print(f"{src.name}: L={L} (NULL included) R={R}; added {need}: " + " ".join(f"U+{ord(c):04X}" for c in add))


if __name__ == "__main__":
    main()
