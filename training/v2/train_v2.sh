#!/usr/bin/env bash
# heb_rivok 2.0 and heb_rivok_lite 2.0, step by step, as they were made. CPU only. See README.md.
#
#   WORK=/path/to/work DATASET=/path/to/line-dataset bash train_v2.sh [STEP ...]
#
# Before the first step:
#   bash ../fetch_inputs.sh "$WORK"                    langdata_lstm files and word lists (the 1.0 inputs)
#   git clone https://huggingface.co/datasets/Rivok/tesseract-heb-rivok-v2-lines "$DATASET"   (git-lfs)
#   python3 -I fetch_v2_sources.py --out "$WORK/v2_sources" --only fonts
#   python3 -I synth/collect_fonts.py "$WORK/v2_sources" "$DATASET" "$WORK/fonts_v2"
# The real lines come from the dataset (pool/); lines/build_lines.py rebuilds them from the scans.
#
# Steps, in order (all of them if none is named):
#   synth     the 61,000 main synthetic lines                               -> $DATA/hebv2-ground-truth
#   link      the 24,643 real lines of the dataset's pool, as r000000.png ...
#   prep      .box/.lstmf for those 85,643 lines, the character set and the model to train (prep_train.py)
#   lists1    heb_rivok's first list (80,932 lines, 25% real; seed 5)
#   train1a   heb_rivok iterations 0-8,000 from the 1.0 average (start model in the dataset)
#   latin     2,000 lines without Hebrew, added to the list (seed 12)
#   train1b   heb_rivok iterations 8,000-60,000
#   rare      3,000 lines carrying the rarest marks (used by the next two lists)
#   stage2    heb_rivok's second list (20,000 lines, 35% real; seed 21), continuing from the 60,000 checkpoint
#   train2    heb_rivok iterations 60,000-80,000 (the list once)
#   lite      heb_rivok_lite's model and list (60,000 lines, no Latin; seed 31)
#   trainlite heb_rivok_lite iterations 0-60,000 from its 1.0 average
#   release   average the last 8 (heb_rivok) / 6 (lite) snapshots, then int8, right-to-left padding and
#             version string, as released; prints the SHA-256 of the four files
# "bash train_v2.sh ours release" skips training: it takes our snapshots from the dataset and rebuilds
# the four released files byte for byte.
# On the 4-core Xeon E5-1630 v4 we used: synth ~5 min (4 jobs), prep ~10 min, heb_rivok 0-60,000 ~3.5 h,
# 60,000-80,000 ~1.5 h, lite ~2 h, each with dev scoring every 2,000 iterations (train/dev_score.py).
#
# Env: PYTHON (Pillow with raqm, numpy, fontTools, fpdf2) [python3 -I], JOBS [4], OMP_THREAD_LIMIT [4],
#      FONTS [$WORK/fonts_v2], DATA [$WORK/v2].
# Exactness: the lines, lists, character sets and run models come out byte for byte as ours (checked;
# README.md). lstmtraining itself is not bit-reproducible across machines and thread counts, so the
# trained weights will differ slightly; the dataset has our snapshots, and "release" run on them gives
# the released files byte for byte.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${WORK:?set WORK to the working folder}"
: "${DATASET:?set DATASET to the line dataset folder}"
WORK=$(cd "$WORK" && pwd); DATASET=$(cd "$DATASET" && pwd)
PYTHON=${PYTHON:-python3 -I}
JOBS=${JOBS:-4}
FONTS=${FONTS:-$WORK/fonts_v2}
DATA=${DATA:-$WORK/v2}
GT=$DATA/hebv2-ground-truth
TEXT=$DATA/synth_text
START=$DATASET/models/start/heb_rivok_1.0_avg5.traineddata
LITE_START=$DATASET/models/start/heb_rivok_lite_1.0_avg5.traineddata
export OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT:-4} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
mkdir -p "$DATA"

GEN="$PYTHON $HERE/synth/gen_lines_v2.py $GT --langdata $WORK/langdata --fonts $FONTS"
VOWELLED="--vowelled tanakh=$TEXT/tanakh_mam_lines_train.txt:8 --vowelled mishnah=$TEXT/mishnah_toratemet_lines.txt:3
          --vowelled poetry=$TEXT/benyehuda_vowelled_lines.txt:4 --vowelled modern=$TEXT/dicta_modern_lines.txt:5"
TRAIN="$PYTHON $HERE/train/train_v2.py --data $DATA --dataset $DATASET"
log() { echo "== $(date '+%F %T') $*"; }

step_synth() {
  mkdir -p "$TEXT"
  for f in "$DATASET"/synth_text/*.txt.gz; do gunzip -c "$f" > "$TEXT/$(basename "$f" .gz)"; done
  # shellcheck disable=SC2086
  $GEN --n 61000 --seed 2026111 --jobs "$JOBS" $VOWELLED
}
step_link()    { $PYTHON "$HERE/train/build_lists.py" --data "$DATA" link "$GT" "$DATASET"; }
step_prep() {
  $PYTHON "$HERE/../prep_train.py" "$GT" --model hebv2 --data-dir "$DATA" --script-dir "$WORK/langdata" \
    --start "$START" --words "$WORK/langdata/hebeng.wordlist" --numbers "$WORK/langdata/hebeng.numbers" \
    --puncs "$WORK/langdata/hebeng.punc" --jobs "$JOBS"
}
step_lists1()  { $PYTHON "$HERE/train/build_lists.py" --data "$DATA" lists hebv2 "$GT"; }
step_train1a() { $TRAIN hebv2 --max 8000 --start-lstm "$DATA/start/hebv2.lstm" --old "$START"; }
step_latin() {
  $GEN --n 2000 --start 61000 --seed 2026112 --jobs "$JOBS" --mix latin=1
  $PYTHON "$HERE/train/lstmf_for.py" "$GT" 61000 63000 --jobs "$JOBS"
  $PYTHON "$HERE/train/build_lists.py" --data "$DATA" add hebv2 "$GT" 61000 63000
}
step_train1b() { $TRAIN hebv2 --max 60000 --start-lstm "$DATA/start/hebv2.lstm" --old "$START"; }
step_rare() {
  # shellcheck disable=SC2086
  $GEN --n 3000 --start 63000 --seed 2026113 --jobs "$JOBS" --mix rare=1 $VOWELLED
  $PYTHON "$HERE/train/lstmf_for.py" "$GT" 63000 66000 --jobs "$JOBS"
}
step_stage2() {
  mkdir -p "$DATA/hebv21/checkpoints"
  cp "$DATA/hebv2/run/snapshots/hebv2_60000.checkpoint" "$DATA/hebv21/checkpoints/hebv21_checkpoint"
  cp "$DATA/hebv2/hebv2.traineddata" "$DATA/hebv21/hebv21.traineddata"
  $PYTHON "$HERE/train/stage_lists.py" --data "$DATA" --gt "$GT" hebv21
}
step_train2()  { $TRAIN hebv21 --offset 60000 --max 80000 --no-stop; }
step_lite() {
  mkdir -p "$DATA/start" "$DATA/heblv2"
  combine_tessdata -u "$LITE_START" "$DATA/start/heblv2" > /dev/null
  # the characters lite 2.0 adds: 11 marks and the em dash (every other character of its lines is in lite 1.0)
  printf '%s\n' "ֱֲֳֵׇֹֻֽֿׁׂ—" > "$DATA/heblv2/new-chars.txt"
  unicharset_extractor --output_unicharset "$DATA/heblv2/my.unicharset" --norm_mode 3 "$DATA/heblv2/new-chars.txt" > /dev/null
  merge_unicharsets "$DATA/start/heblv2.lstm-unicharset" "$DATA/heblv2/my.unicharset" "$DATA/heblv2/unicharset" > /dev/null
  combine_lang_model --input_unicharset "$DATA/heblv2/unicharset" --script_dir "$WORK/langdata" \
    --numbers "$WORK/langdata/heb.numbers" --puncs "$WORK/langdata/heb.punc" --words "$WORK/langdata/heb.wordlist" \
    --output_dir "$DATA" --pass_through_recoder --lang_is_rtl --lang heblv2 > /dev/null
  $PYTHON "$HERE/train/stage_lists.py" --data "$DATA" --gt "$GT" heblv2 --size 60000 --latin-share 0 --hq-repeat \
    --charset "$DATA/heblv2/unicharset" --seed 31
}
step_trainlite() { $TRAIN heblv2 --max 60000 --start-lstm "$DATA/start/heblv2.lstm" --old "$LITE_START"; }

build() {   # NAME RUN K PROTO VERSION: average, int8, right-to-left padding, version string
  local name=$1 run=$2 k=$3 proto=$4 version=$5 out=$DATA/release
  mkdir -p "$out/tessdata_best" "$out/tessdata"
  $PYTHON "$HERE/train/average.py" --data "$DATA" "$run" --k "$k" --out "$out/tessdata_best/$name.traineddata"
  combine_tessdata -e "$out/tessdata_best/$name.traineddata" "$out/$name.lstm" > /dev/null
  lstmtraining --stop_training --convert_to_int --continue_from "$out/$name.lstm" --traineddata "$proto" \
    --model_output "$out/tessdata/$name.traineddata" > /dev/null
  rm -f -- "${out:?}/${name:?}.lstm"
  for d in tessdata_best tessdata; do
    $PYTHON "$HERE/../pad_rtl.py" "$out/$d/$name.traineddata" "$out/$d/$name.traineddata"
    printf '%s' "$version" > "$out/$d/$name.version"
    combine_tessdata -o "$out/$d/$name.traineddata" "$out/$d/$name.version" > /dev/null
    rm -f -- "${out:?}/${d:?}/${name:?}.version" "${out:?}/${d:?}/${name:?}.traineddata.__tmp__"
  done
}
step_ours() {   # instead of training: put our snapshots and run models where "release" reads them
  mkdir -p "$DATA/hebv21/run/float" "$DATA/heblv2/run/float"
  for i in 66000 68000 70000 72000 74000 76000 78000 80000; do
    ln -sf "$DATASET/models/runs/heb_rivok/stage2_$i.traineddata" "$DATA/hebv21/run/float/hebv21_$i.traineddata"
  done
  for i in 50000 52000 54000 56000 58000 60000; do
    ln -sf "$DATASET/models/runs/heb_rivok_lite/lite_$i.traineddata" "$DATA/heblv2/run/float/heblv2_$i.traineddata"
  done
  ln -sf "$DATASET/models/runs/heb_rivok/heb_rivok_v2.traineddata" "$DATA/hebv21/hebv21.traineddata"
  ln -sf "$DATASET/models/runs/heb_rivok_lite/heb_rivok_lite_v2.traineddata" "$DATA/heblv2/heblv2.traineddata"
}
step_release() {
  build heb_rivok hebv21 8 "$DATA/hebv21/hebv21.traineddata" \
    "heb_rivok 2.0 (2026-11-13, Rivok Labs): tessdata_best script/Hebrew fine-tuned on real old-print lines (Hebrew Wikisource) and synthetic Hebrew, Hebrew/English and vowelled lines; 23 unused Hebrew characters added to the unicharset so Tesseract lays pages out right to left"
  build heb_rivok_lite heblv2 6 "$DATA/heblv2/heblv2.traineddata" \
    "heb_rivok_lite 2.0 (2026-11-13, Rivok Labs): tessdata_best heb fine-tuned on real old-print lines (Hebrew Wikisource) and synthetic Hebrew and vowelled lines; Hebrew only, use with +eng"
  (cd "$DATA/release" && sha256sum tessdata_best/*.traineddata tessdata/*.traineddata)
  echo "released 2.0: tessdata_best/heb_rivok 673a4193..., heb_rivok_lite 3e70e4bd...; tessdata/heb_rivok 09ef1830..., heb_rivok_lite 59da073a..."
}

ALL="synth link prep lists1 train1a latin train1b rare stage2 train2 lite trainlite release"
for s in ${@:-$ALL}; do
  case " $ALL ours " in *" $s "*) ;; *) echo "unknown step $s (steps: $ALL; or: ours release)" >&2; exit 2 ;; esac
  log "$s"
  "step_$s"
done
log done
