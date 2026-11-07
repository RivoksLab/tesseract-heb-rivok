#!/usr/bin/env bash
# Rerun every public benchmark number of the model card (about 20 minutes on 2 CPU cores).
#
#   bash run_all.sh
#
# Env:
#   HZ        HaZefira set (built by `hazefira.py build` if missing)  [eval/data/hazefira]
#   GLOT_DIR  local folder for the GlotOCR parquet (default: Hugging Face cache)
#   STOCK     tessdata folder with the stock heb + eng (fetch_models.sh)  [Tesseract's own]
#   RESULTS   where predictions and scores go                         [eval/results]
#   JOBS      Tesseract processes in parallel, one thread each          [2]
#   PYTHON    Python with pyarrow, huggingface_hub, Pillow, python-bidi [python3 -I]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TD=$HERE/../tessdata
HZ=${HZ:-$HERE/data/hazefira}
RESULTS=${RESULTS:-$HERE/results}
JOBS=${JOBS:-2}
PYTHON=${PYTHON:-python3 -I}
STOCK_TD=(); [ -n "${STOCK:-}" ] && STOCK_TD=(--tessdata "$STOCK")
GLOT_DD=(); [ -n "${GLOT_DIR:-}" ] && GLOT_DD=(--data-dir "$GLOT_DIR")
mkdir -p "$RESULTS"
run() { local name=$1; shift; echo "== $name"; $PYTHON "$@" --jobs "$JOBS" --out "$RESULTS/$name.jsonl" | tee "$RESULTS/$name.txt"; }

[ -f "$HZ/lines.tsv" ] || $PYTHON "$HERE/hazefira.py" build --data-dir "$HZ"

run glot_heb_rivok       "$HERE/glotocr.py" "${GLOT_DD[@]}" --tessdata "$TD" --lang heb_rivok
run glot_heb_rivok_lite  "$HERE/glotocr.py" "${GLOT_DD[@]}" --tessdata "$TD" --lang heb_rivok_lite
run glot_ubuntu_heb      "$HERE/glotocr.py" "${GLOT_DD[@]}" "${STOCK_TD[@]}" --lang heb

run hz_lines_heb_rivok       "$HERE/hazefira.py" lines --data-dir "$HZ" --tessdata "$TD" --lang heb_rivok
run hz_lines_heb_rivok_lite  "$HERE/hazefira.py" lines --data-dir "$HZ" --tessdata "$TD" --lang heb_rivok_lite
run hz_lines_ubuntu_heb      "$HERE/hazefira.py" lines --data-dir "$HZ" "${STOCK_TD[@]}" --lang heb

run hz_pages_heb_rivok       "$HERE/hazefira.py" pages --data-dir "$HZ" --tessdata "$TD" --lang heb_rivok
run hz_pages_ubuntu_heb_eng  "$HERE/hazefira.py" pages --data-dir "$HZ" "${STOCK_TD[@]}" --lang heb+eng
