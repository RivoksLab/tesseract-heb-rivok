#!/usr/bin/env bash
# Download the stock models the published numbers compare against, for systems that do not
# have Ubuntu's packages. Ubuntu's `tesseract-ocr-heb` and `tesseract-ocr-eng` are exactly
# tessdata_fast heb and eng (checked byte for byte), so on Ubuntu/Debian this is not needed.
#
#   bash fetch_models.sh [DIR]        (default: eval/models/stock)
#   then: --tessdata DIR --lang heb   (pages: --lang heb+eng)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DIR=${1:-$HERE/models/stock}
FAST_REV=87416418657359cb625c412a48b6e1d6d41c29bd      # tesseract-ocr/tessdata_fast
mkdir -p "$DIR"
for f in heb eng; do
  [ -s "$DIR/$f.traineddata" ] || curl -fsSL --retry 3 -o "$DIR/$f.traineddata" \
    "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/$FAST_REV/$f.traineddata"
done
cd "$DIR"
sha256sum -c - <<'EOF'
11f9e43ab227f786352a50f75c94c2e9906f1baba86d93276da19da7ce0904db  heb.traineddata
7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2  eng.traineddata
EOF
