#!/usr/bin/env bash
# Download the training inputs into WORK and build the Hebrew+English dictionary lists.
#
#   bash fetch_inputs.sh WORK
#
# WORK/langdata/  langdata_lstm files (Apache-2.0), pinned to the commit used for training:
#                 heb.{training_text,wordlist,numbers,punc}, eng.{training_text,wordlist,numbers,punc},
#                 {Hebrew,Latin}.{unicharset,xheights}, radical-stroke.txt
#                 + hebeng.{wordlist,numbers,punc}, the dictionary lists of heb_rivok (built here)
# WORK/start/     start models from tessdata_best (Apache-2.0):
#                 heb.traineddata (for heb_rivok_lite), Hebrew.traineddata = script/Hebrew (for heb_rivok)
# Every file is checked against inputs.sha256. Fonts: fetch_fonts.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${1:?usage: fetch_inputs.sh WORK}
LANGDATA_REV=07930fd9f246622c26eb5de794d9212ceac432d3     # tesseract-ocr/langdata_lstm, 2024-03-09
BEST_REV=e12c65a915945e4c28e237a9b52bc4a8f39a0cec         # tesseract-ocr/tessdata_best
mkdir -p "$WORK/langdata" "$WORK/start"
WORK=$(cd "$WORK" && pwd)

get() {   # URL DEST: download unless DEST is already there
  [ -s "$2" ] && return 0
  echo "  $1"
  curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

echo "langdata_lstm @ $LANGDATA_REV"
for f in heb/heb.training_text heb/heb.wordlist heb/heb.numbers heb/heb.punc \
         eng/eng.training_text eng/eng.wordlist eng/eng.numbers eng/eng.punc \
         Hebrew.unicharset Hebrew.xheights Latin.unicharset Latin.xheights radical-stroke.txt; do
  get "https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/$LANGDATA_REV/$f" "$WORK/langdata/$(basename "$f")"
done

echo "tessdata_best @ $BEST_REV"
get "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/$BEST_REV/heb.traineddata" "$WORK/start/heb.traineddata"
get "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/$BEST_REV/script/Hebrew.traineddata" "$WORK/start/Hebrew.traineddata"

echo "building hebeng.{wordlist,numbers,punc}"
cd "$WORK/langdata"
# wordlist: Tesseract's Hebrew word list, then the first 150,000 English entries that are plain
# ASCII letters (eng.wordlist is in frequency order)
python3 -I - <<'EOF'
heb = open("heb.wordlist", encoding="utf-8").read()
eng = [w for w in (l.strip() for l in open("eng.wordlist", encoding="utf-8")) if w and w.isascii() and w.isalpha()][:150000]
with open("hebeng.wordlist", "w", encoding="utf-8") as f:
    f.write(heb + "".join(w + "\n" for w in eng))
EOF
# numbers and punctuation patterns: both languages' lists, sorted and de-duplicated. The order
# is that of `sort` in the en_US.UTF-8 locale, which is what the released model used; another
# locale gives the same patterns in another order (and a checksum warning below).
if locale -a 2>/dev/null | grep -qiE '^en_US\.utf-?8$'; then SORT_LOCALE=en_US.UTF-8; else SORT_LOCALE=C.UTF-8
  echo "  warning: locale en_US.UTF-8 not installed; sorting with C.UTF-8"; fi
LC_ALL=$SORT_LOCALE sort -u heb.numbers eng.numbers > hebeng.numbers
LC_ALL=$SORT_LOCALE sort -u heb.punc eng.punc > hebeng.punc

echo "checking sha256"
cd "$WORK"
if sha256sum -c --quiet "$HERE/inputs.sha256"; then
  echo "all inputs match the ones used for heb_rivok and heb_rivok_lite"
else
  echo "warning: some files differ from the ones used for training (see above)" >&2
  exit 1
fi
