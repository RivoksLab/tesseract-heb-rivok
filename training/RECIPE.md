# Training recipe

How `heb_rivok` and `heb_rivok_lite` were made, with the commands to make them again. Both
are fine-tunes of Google's tessdata_best models on synthetic text lines; nothing is trained
from scratch and no real document is used.

| model | start model | lines | iterations | generator | dictionary |
|---|---|---:|---:|---|---|
| `heb_rivok` (round 4) | tessdata_best `script/Hebrew` (48 px, Hebrew + Latin) | 60,000 | 20,000 | `--mixed`, seed 4 | Hebrew + 150,000 English words |
| `heb_rivok_lite` (round 3) | tessdata_best `heb` (36 px, Hebrew only) | 50,000 | 15,000 | `--legacy-r3`, seed 3 | Hebrew |

Rounds 1 and 2 were earlier experiments with the same pipeline; their models are not released.

## Files

| file | what it does |
|---|---|
| `fetch_inputs.sh` | downloads langdata_lstm text and word lists and the two start models at pinned commits, builds the Hebrew+English lists, checks everything against `inputs.sha256` |
| `fonts.txt` | the 50 fonts, in the order the generator uses them, with licence, SHA-256 and source |
| `fetch_fonts.sh` | downloads / copies the fonts of `fonts.txt` into one folder and checks them |
| `gen_lines.py` | renders synthetic line images + ground truth (`NNNNNN.png`, `NNNNNN.gt.txt`) |
| `prep_train.py` | box files, `.lstmf` files, train/eval lists, unicharset, proto model (tesstrain's recipes without its Makefile) |
| `train.sh` | runs the four steps: generate, prep, `lstmtraining`, convert (float + int8), and stamps the version string |

## Environment

The released models were trained on Ubuntu 26.04, x86-64 with AVX2/FMA, CPU only.

- **Tesseract 5.5.0** with its training tools (Ubuntu package `tesseract-ocr` 5.5.0-1build1;
  `tesseract --version`: tesseract 5.5.0, leptonica-1.86.0, Found AVX2/AVX/FMA/SSE4.1,
  OpenMP 201511). `prep_train.py` runs `tesseract … lstm.train`, which needs Tesseract's own
  `configs/lstm.train` and `eng.traineddata` (`apt install tesseract-ocr tesseract-ocr-eng`).
  Tools used: `tesseract`, `combine_tessdata`, `unicharset_extractor`, `merge_unicharsets`,
  `combine_lang_model`, `lstmtraining`.
- **tesstrain: not needed.** `prep_train.py` does what tesstrain's Makefile does for
  `LANG_TYPE=RTL` with a start model (reference: tesstrain commit `405346a3`, 2025-04-18), run
  directly and in parallel; with tens of thousands of lines `make` spent minutes resolving
  pattern rules. Two deliberate differences, both explained in `prep_train.py`: the box text
  keeps brackets unmirrored inside Hebrew words, and every line with Hebrew gets an explicit
  right-to-left direction.
- **Python 3.12** (3.12.16) with Pillow 13.0.0 (the PyPI wheel; it bundles FreeType 2.14.3,
  HarfBuzz 14.5.0, raqm 0.11.0, FriBiDi 1.0.16), numpy 2.5.4, fontTools 4.68.0, fpdf2 2.9.0:
  `pip install pillow==13.0.0 numpy==2.5.4 fonttools==4.68.0 fpdf2==2.9.0`. Pillow must have
  raqm: `python -c "from PIL import features; print(features.check('raqm'))"` prints True.
- `curl`, `sha256sum`, GNU `sort` and the `en_US.UTF-8` locale (only for the order of two
  dictionary pattern files, see `fetch_inputs.sh`).
- Disk: about 3 GB for round 4 (2.4 GB of images, boxes and `.lstmf` files, 0.5 GB of
  checkpoints), about 2 GB for round 3.

## 1. Inputs

```bash
export WORK=/path/to/work          # ~6 GB free for both rounds; nothing is written elsewhere
bash training/fetch_inputs.sh "$WORK"
bash training/fetch_fonts.sh "$WORK/fonts"
```

`fetch_inputs.sh` fills:
- `$WORK/langdata/`: from [tesseract-ocr/langdata_lstm](https://github.com/tesseract-ocr/langdata_lstm)
  at `07930fd9` (Apache-2.0): `heb.training_text` (Tesseract's Hebrew training text),
  `heb.wordlist`, `heb.numbers`, `heb.punc`, the same four for `eng`, `Hebrew.unicharset`,
  `Hebrew.xheights`, `Latin.unicharset`, `Latin.xheights`, `radical-stroke.txt`; and the
  dictionary lists of `heb_rivok`, built from them:
  - `hebeng.wordlist`: `heb.wordlist`, then the first 150,000 entries of `eng.wordlist` (which
    is in frequency order) that consist of ASCII letters only;
  - `hebeng.numbers`, `hebeng.punc`: `LC_ALL=en_US.UTF-8 sort -u heb.X eng.X`.
- `$WORK/start/heb.traineddata` and `$WORK/start/Hebrew.traineddata`: tessdata_best `heb` and
  `script/Hebrew` from [tesseract-ocr/tessdata_best](https://github.com/tesseract-ocr/tessdata_best)
  at `e12c65a9` (Apache-2.0).

`fetch_fonts.sh` downloads the Google Fonts files from
[google/fonts](https://github.com/google/fonts) at the commits named in `fonts.txt` and copies
DejaVu, FreeFont and Liberation from their Debian/Ubuntu packages (`apt install
fonts-dejavu-core fonts-freefont-ttf fonts-liberation`). No font is distributed here.

## 2. heb_rivok (round 4)

```bash
WORK=$WORK START=$WORK/start/Hebrew.traineddata SEED=4 GEN_ARGS=--mixed \
  WORDS=$WORK/langdata/hebeng.wordlist NUMBERS=$WORK/langdata/hebeng.numbers PUNCS=$WORK/langdata/hebeng.punc \
  VERSION='heb_rivok 1.0.1 (2026-11-07, Rivok Labs): tessdata_best script/Hebrew fine-tuned on synthetic Hebrew and Hebrew/English lines; 23 unused Hebrew characters added to the unicharset so Tesseract lays pages out right to left' \
  bash training/train.sh heb_rivok 60000 20000
```

What `train.sh` runs, step by step (`$D` = `$WORK/data`):

```bash
# 1. 60,000 synthetic lines; line i is seeded with 4 * 1000003 + i
python3 -I gen_lines.py $D/heb_rivok-ground-truth --langdata $WORK/langdata \
  --fonts fonts.txt --font-dir $WORK/fonts --n 60000 --seed 4 --mixed --jobs 6

# 2. boxes, lstmf, lists (90/10, at most 1,000 eval lines, shuffle seed 20261106),
#    unicharset (155 -> 160 characters), proto model; script files: Hebrew + Latin
python3 -I prep_train.py $D/heb_rivok-ground-truth --model heb_rivok --data-dir $D \
  --script-dir $D/scripts-heb_rivok --start $WORK/start/Hebrew.traineddata \
  --words $WORK/langdata/hebeng.wordlist --numbers $WORK/langdata/hebeng.numbers \
  --puncs $WORK/langdata/hebeng.punc --jobs 6

# 3. fine-tune (OMP_THREAD_LIMIT=4)
lstmtraining --debug_interval 0 \
  --traineddata $D/heb_rivok/heb_rivok.traineddata \
  --old_traineddata $WORK/start/Hebrew.traineddata \
  --continue_from $D/start/heb_rivok.lstm \
  --learning_rate 0.0001 \
  --model_output $D/heb_rivok/checkpoints/heb_rivok \
  --train_listfile $D/heb_rivok/list.train --eval_listfile $D/heb_rivok/list.eval \
  --max_iterations 20000 --target_error_rate 0.01

# 4. float (tessdata_best) and int8 (tessdata) models from the last checkpoint
lstmtraining --stop_training --continue_from $D/heb_rivok/checkpoints/heb_rivok_checkpoint \
  --traineddata $D/heb_rivok/heb_rivok.traineddata --model_output $WORK/models/tessdata_best/heb_rivok.traineddata
lstmtraining --stop_training --convert_to_int --continue_from $D/heb_rivok/checkpoints/heb_rivok_checkpoint \
  --traineddata $D/heb_rivok/heb_rivok.traineddata --model_output $WORK/models/tessdata/heb_rivok.traineddata

# 5. lay pages out right to left: heb_rivok knows the Latin letters, so without this Tesseract
#    reads multi-column pages left column first (model card, Limits). Network unchanged.
python3 training/pad_rtl.py $WORK/models/tessdata/heb_rivok.traineddata $WORK/models/tessdata/heb_rivok.traineddata
python3 training/pad_rtl.py $WORK/models/tessdata_best/heb_rivok.traineddata $WORK/models/tessdata_best/heb_rivok.traineddata

# 6. version string (no trailing newline), in both files
printf '%s' "$VERSION" > heb_rivok.version
combine_tessdata -o $WORK/models/tessdata/heb_rivok.traineddata heb_rivok.version
combine_tessdata -o $WORK/models/tessdata_best/heb_rivok.traineddata heb_rivok.version
rm heb_rivok.version $WORK/models/tessdata*/heb_rivok.traineddata.__tmp__   # backups left by -o
```

`--old_traineddata` makes `lstmtraining` map the start model's 155 output characters onto the
new 160 (adds ־ ׳ ״ – •) and keep everything else of the network. The original run took
2 h 35 min on 4 threads while other jobs shared the CPU. `lstmtraining` sometimes aborts
after printing `Finished!` (its background evaluation thread races the shutdown);
`train.sh` therefore judges success by the log.

## 3. heb_rivok_lite (round 3)

```bash
WORK=$WORK START=$WORK/start/heb.traineddata SEED=3 GEN_ARGS=--legacy-r3 SCRIPTS=Hebrew \
  VERSION='heb_rivok_lite 1.0 (2026-11-07, Rivok Labs): tessdata_best heb fine-tuned on synthetic Hebrew lines; Hebrew only, use with +eng' \
  bash training/train.sh heb_rivok_lite 50000 15000
```

Same steps with these differences:
- **Generator as it was for round 3: `--legacy-r3`.** Round 4 added two fixes to
  `gen_lines.py` (ink thinning is skipped when it would erase a light font, and a line that
  comes out nearly blank is degraded again; round 3 has ~3% blank images with text labels) and
  a different test for empty neighbour lines. `--legacy-r3` switches all three back. The
  default profile (no `--mixed`) is round 3's text mix and character set: Hebrew, digits,
  punctuation, ₪ € & @ _ and six niqqud marks, no Latin letters.
- **Script files: Hebrew only (`SCRIPTS=Hebrew`).** `combine_lang_model` reads
  `Latin.unicharset` even for a Hebrew-only character set (Tesseract's internal "Joined" entry
  is Latin) and it changes character properties in the unicharset, so round 3, made before
  the Latin files were added, needs them absent to come out identical.
- Start model tessdata_best `heb`; dictionary `heb.wordlist`, `heb.numbers`, `heb.punc` (the
  `train.sh` defaults); unicharset 69 -> 78; 50,000 lines, 15,000 iterations (about 25 min).

## What has been checked

Against the files of the original runs (2026-11-07):
- every downloaded input matches `inputs.sha256`, and the 50 fonts match `fonts.txt`;
- `gen_lines.py` reproduces the original training images and labels **byte for byte**
  (200-line samples of each round: `--mixed --seed 4` for round 4, `--legacy-r3 --seed 3` for
  round 3; without `--legacy-r3`, 21 of the 200 round-3 images differ);
- `prep_train.py` reproduces the original `.box` and `.lstmf` files byte for byte (20-line
  samples), and its unicharset and `combine_lang_model` steps, run on the full ground-truth
  text of each round with these inputs, give the original unicharset and proto model
  (dictionaries, recoder) byte for byte;
- steps 4-5 applied to the original trained checkpoints' models give the released files byte
  for byte:

| file | SHA-256 |
|---|---|
| `tessdata/heb_rivok.traineddata` | `ae48e574583afa82832704d402a17cc496ecfd9d58ef29cc92a124de57d9bda8` |
| `tessdata_best/heb_rivok.traineddata` | `706ad13c68235c1832b7d19ac3925ee3fc105eba2da83181a0f810b9c39c35ce` |
| `tessdata/heb_rivok_lite.traineddata` | `a92584816adf5c9ae38afc255e3530477516318c21c2e711fd57b05535239360` |
| `tessdata_best/heb_rivok_lite.traineddata` | `baceb6c52bf69550ffae52acad9297619a1b7fc8553890caf299798f7348e4cd` |

`lstmtraining` itself was not run again for this release (it takes hours of CPU), and we have
not checked whether it is bit-for-bit deterministic across thread counts and CPUs. Expect a
retrained model to match closely in accuracy rather than byte for byte. Rendering can also
differ with other versions of Pillow, FreeType, HarfBuzz or the fonts.

To check a retrained model, see `../eval/`.
