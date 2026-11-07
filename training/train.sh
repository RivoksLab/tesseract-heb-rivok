#!/usr/bin/env bash
# Fine-tune a tessdata_best Hebrew model on synthetic lines. CPU only. See RECIPE.md.
#
#   WORK=/path/to/work START=$WORK/start/Hebrew.traineddata bash train.sh MODEL N_LINES MAX_ITERATIONS
#
# WORK must hold langdata/ and start/ (fetch_inputs.sh) and fonts/ (fetch_fonts.sh).
# Env:
#   START       start model (a float tessdata_best .traineddata)                     [required]
#   SEED        generator seed (round 3: 3, round 4: 4)                               [20261106]
#   GEN_ARGS    extra gen_lines.py flags (round 4: --mixed; round 3: --legacy-r3)    []
#   SCRIPTS     langdata_lstm scripts for combine_lang_model (round 3: Hebrew)     [Hebrew Latin]
#   WORDS NUMBERS PUNCS   dictionary lists for the model     [$WORK/langdata/heb.{wordlist,numbers,punc}]
#   VERSION     version string stamped into both output files (combine_tessdata -o) [not stamped]
#   JOBS        parallel workers for generation and prep                            [6]
#   PYTHON      Python command (Pillow with raqm, numpy, fontTools, fpdf2)           [python3 -I]
#   OMP_THREAD_LIMIT  lstmtraining threads                                           [4]
#   FONTS FONT_DIR    font list and the folder holding the fonts         [training/fonts.txt, $WORK/fonts]
# Steps:
#   1. gen_lines.py   -> $WORK/data/MODEL-ground-truth   (skipped if it is already complete)
#   2. prep_train.py  -> lstmf files, lists, unicharset, proto model (tesstrain's recipes, no make)
#   3. lstmtraining   fine-tune from START, learning rate 1e-4
#   4. float model    -> $WORK/models/tessdata_best/MODEL.traineddata
#      int8 model     -> $WORK/models/tessdata/MODEL.traineddata (--convert_to_int)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${WORK:?set WORK to the working folder}"
: "${START:?set START to the start .traineddata}"
WORK=$(cd "$WORK" && pwd)
MODEL=${1:?usage: train.sh MODEL N_LINES MAX_ITERATIONS}
N=${2:?usage: train.sh MODEL N_LINES MAX_ITERATIONS}
ITERS=${3:?usage: train.sh MODEL N_LINES MAX_ITERATIONS}
PYTHON=${PYTHON:-python3 -I}
JOBS=${JOBS:-6}
FONTS=${FONTS:-$HERE/fonts.txt}
FONT_DIR=${FONT_DIR:-$WORK/fonts}
WORDS=${WORDS:-$WORK/langdata/heb.wordlist}; NUMBERS=${NUMBERS:-$WORK/langdata/heb.numbers}; PUNCS=${PUNCS:-$WORK/langdata/heb.punc}
DATA=$WORK/data
GT=$DATA/$MODEL-ground-truth
# OpenMP threads that spin-wait collapse to <1 iteration/s when other jobs share the CPU
export OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT:-4} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
log() { echo "[$(date +%T)] $*"; }

# script files for combine_lang_model. They set character properties in the unicharset, and
# Latin.unicharset is read even for a Hebrew-only character set (Tesseract's "Joined" entry is
# Latin), so round 3 (heb_rivok_lite, SCRIPTS=Hebrew) must not see it to come out identical.
SCRIPT_DIR=$DATA/scripts-$MODEL
rm -rf -- "${SCRIPT_DIR:?}"
mkdir -p "$SCRIPT_DIR" "$WORK/logs"
cp "$WORK/langdata/radical-stroke.txt" "$SCRIPT_DIR/"
for s in ${SCRIPTS:-Hebrew Latin}; do
  cp "$WORK/langdata/$s.unicharset" "$WORK/langdata/$s.xheights" "$SCRIPT_DIR/"
done

if [ ! -f "$GT/.done" ]; then
  log "generating $N lines"
  # shellcheck disable=SC2086
  $PYTHON "$HERE/gen_lines.py" "$GT" --langdata "$WORK/langdata" --fonts "$FONTS" --font-dir "$FONT_DIR" \
    --n "$N" --jobs "$JOBS" --seed "${SEED:-20261106}" ${GEN_ARGS:-}
  touch "$GT/.done"
fi

log "prep: boxes, lstmf, lists, unicharset, proto model"
$PYTHON "$HERE/prep_train.py" "$GT" --model "$MODEL" --data-dir "$DATA" --script-dir "$SCRIPT_DIR" \
  --jobs "$JOBS" --start "$START" --words "$WORDS" --numbers "$NUMBERS" --puncs "$PUNCS"

log "training to $ITERS iterations"
mkdir -p "$DATA/$MODEL/checkpoints"
# lstmtraining can abort AFTER "Finished!" (its background eval thread races the shutdown;
# one run died with a PNG assert although every image decodes). Judge success by the log.
TRAIN_LOG=$WORK/logs/$MODEL.train.log
set +e
lstmtraining --debug_interval 0 \
  --traineddata "$DATA/$MODEL/$MODEL.traineddata" \
  --old_traineddata "$START" \
  --continue_from "$DATA/start/$MODEL.lstm" \
  --learning_rate 0.0001 \
  --model_output "$DATA/$MODEL/checkpoints/$MODEL" \
  --train_listfile "$DATA/$MODEL/list.train" \
  --eval_listfile "$DATA/$MODEL/list.eval" \
  --max_iterations "$ITERS" --target_error_rate 0.01 2>&1 | grep --line-buffered -v "^Loaded" | tee "$TRAIN_LOG"
set -e
grep -q "^Finished!" "$TRAIN_LOG" || { log "training did not finish"; exit 1; }

log "converting"
OUT=$WORK/models
mkdir -p "$OUT/tessdata_best" "$OUT/tessdata"
lstmtraining --stop_training --continue_from "$DATA/$MODEL/checkpoints/${MODEL}_checkpoint" \
  --traineddata "$DATA/$MODEL/$MODEL.traineddata" --model_output "$OUT/tessdata_best/$MODEL.traineddata"
lstmtraining --stop_training --convert_to_int --continue_from "$DATA/$MODEL/checkpoints/${MODEL}_checkpoint" \
  --traineddata "$DATA/$MODEL/$MODEL.traineddata" --model_output "$OUT/tessdata/$MODEL.traineddata"

# Tesseract lays a page out right to left only when the unicharset has more right-to-left than
# left-to-right characters. A model that knows the Latin letters needs padding (model card, Limits);
# pad_rtl.py leaves a model that is already right-to-left unchanged.
for d in "$OUT/tessdata_best" "$OUT/tessdata"; do
  $PYTHON "$HERE/pad_rtl.py" "$d/$MODEL.traineddata" "$d/$MODEL.traineddata"
done

if [ -n "${VERSION:-}" ]; then
  log "version string: $VERSION"
  for d in "$OUT/tessdata_best" "$OUT/tessdata"; do
    printf '%s' "$VERSION" > "$d/$MODEL.version"          # no trailing newline
    combine_tessdata -o "$d/$MODEL.traineddata" "$d/$MODEL.version"
    rm -f -- "${d:?}/$MODEL.version" "${d:?}/$MODEL.traineddata.__tmp__"   # combine_tessdata -o leaves a backup
  done
fi
log "done: $OUT/tessdata/$MODEL.traineddata (int8), $OUT/tessdata_best/$MODEL.traineddata (float)"
