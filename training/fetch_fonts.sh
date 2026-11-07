#!/usr/bin/env bash
# Put the 50 training fonts of fonts.txt into one folder, under the names gen_lines.py expects.
#
#   bash fetch_fonts.sh FONT_DIR [fonts.txt]
#
# Google Fonts files are downloaded from the google/fonts repository at the pinned commit in each
# URL. DejaVu, FreeFont and Liberation are copied from their Debian/Ubuntu packages
# (apt install fonts-dejavu-core fonts-freefont-ttf fonts-liberation). Every file is checked
# against the sha256 column: a different version still works, but gives different images.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:?usage: fetch_fonts.sh FONT_DIR [fonts.txt]}
LIST=${2:-$HERE/fonts.txt}
mkdir -p "$DEST"
missing=0; differ=0
while IFS=$'\t' read -r file family licence sha source; do
  case "$file" in ''|'#'*) continue ;; esac
  out=$DEST/$file
  if [ ! -s "$out" ]; then
    case "$source" in
      http*) curl -fsSL --retry 3 -o "$out.part" "$source" && mv "$out.part" "$out" ;;
      /*)    src=${source%% *}
             if [ -f "$src" ]; then cp "$src" "$out"
             else echo "missing: $file: not at $src; install the package named in fonts.txt"; missing=$((missing+1)); continue; fi ;;
      *)     echo "missing: $file: unknown source '$source'"; missing=$((missing+1)); continue ;;
    esac
  fi
  got=$(sha256sum "$out" | cut -d' ' -f1)
  if [ "$got" != "$sha" ]; then echo "differs: $file ($family): sha256 $got, training used $sha"; differ=$((differ+1)); fi
done < "$LIST"
n=$(grep -cv '^#' "$LIST")
echo "$((n - missing)) of $n fonts in $DEST; $differ differ from the training copies"
[ "$missing" -eq 0 ]
