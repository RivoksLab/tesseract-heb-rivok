#!/usr/bin/env bash
# Build the tesseract-heb-rivok wheel and sdist from the repository's tessdata/ (the "fast" models).
# The model files are copied in and checked against their published SHA-256 first.
#   PYTHON=/path/to/python ./build.sh      # needs the `build` package and setuptools>=77
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p src/tesseract_heb_rivok/tessdata
for m in heb_rivok heb_rivok_lite; do
  cp ../tessdata/$m.traineddata src/tesseract_heb_rivok/tessdata/
done
cp ../LICENSE ../NOTICE .
"$PY" - <<'EOF'
import hashlib, pathlib, sys
sys.path.insert(0, "src")
from tesseract_heb_rivok import MODELS
for name, (_, sha) in MODELS.items():
    got = hashlib.sha256(pathlib.Path(f"src/tesseract_heb_rivok/tessdata/{name}.traineddata").read_bytes()).hexdigest()
    if got != sha:
        sys.exit(f"{name}: SHA-256 {got} does not match {sha}")
print("model files match their published SHA-256")
EOF
rm -rf dist build src/*.egg-info
"$PY" -m build --no-isolation
ls -l dist
