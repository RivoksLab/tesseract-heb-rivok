# tesseract-heb-rivok

[heb_rivok](https://github.com/RivoksLab/tesseract-heb-rivok) and heb_rivok_lite are
Tesseract 5 models for printed Hebrew, fine-tuned by [Rivok Labs](https://rivoklabs.com).
This package installs them with pip and tells Tesseract where they are. You still need
Tesseract itself (`apt install tesseract-ocr`, `brew install tesseract`, or the Windows
installer).

```bash
pip install tesseract-heb-rivok
```

| model | for | version |
|---|---|---|
| `heb_rivok` | Hebrew documents with English words and numbers. English is in the model, so no `+eng` | 1.0.1 |
| `heb_rivok_lite` | Hebrew only, at the speed of the standard `heb` model; add `+eng` for English | 1.0 |

The files are the integer ("fast") models from the
[GitHub release](https://github.com/RivoksLab/tesseract-heb-rivok/releases). The float
models for further fine-tuning are there too.

## Use

`python -m tesseract_heb_rivok` prints the folder that holds the two models:

```bash
tesseract page.png out --tessdata-dir "$(python -m tesseract_heb_rivok)" -l heb_rivok
```

From Python with [pytesseract](https://pypi.org/project/pytesseract/):

```python
import pytesseract
import tesseract_heb_rivok

text = pytesseract.image_to_string(
    "page.png", lang="heb_rivok",
    config=f'--tessdata-dir "{tesseract_heb_rivok.tessdata_dir()}"')
```

**When Tesseract needs more than these two models.** Some uses need Tesseract's own files
as well: another language (`-l heb_rivok_lite+eng`), page rotation (`osd`) or output
configs (`hocr`, `pdf`). OCRmyPDF and Docling need them too. For these, use the merged
folder: `--merged` builds a folder in your cache that links these models and everything in
your system's tessdata.

```bash
export TESSDATA_PREFIX="$(python -m tesseract_heb_rivok --merged)"
ocrmypdf -l heb_rivok scanned.pdf searchable.pdf
tesseract page.png out -l heb_rivok_lite+eng hocr
```

[Docling](https://github.com/docling-project/docling), with its Tesseract command-line
engine:

```python
import os
import tesseract_heb_rivok
from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, TesseractCliOcrOptions

tessdata = tesseract_heb_rivok.merged_tessdata_dir()
os.environ["TESSDATA_PREFIX"] = tessdata          # Docling lists languages without --tessdata-dir
opts = PdfPipelineOptions()
opts.ocr_options = TesseractCliOcrOptions(lang=["heb_rivok"], path=tessdata, mode=OcrMode.FULL_PAGE)
```

`python -m tesseract_heb_rivok --check` verifies the files and that Tesseract loads them.

## Results

Character error rate, lower is better; `fast` models, Tesseract 5.5.0.

| test | `heb` (Ubuntu/Debian) | heb_rivok_lite | **heb_rivok** |
|---|---:|---:|---:|
| 403 line crops from real Hebrew documents | 4.7% | 1.9% | **1.4%** |
| [GlotOCR-bench](https://huggingface.co/datasets/cis-lmu/GlotOCR-bench) Hebrew, clean images | 21.2% | 20.6% | **18.5%** |
| [HaZefira](https://github.com/omilab/historical_press) newspaper, 4,013 line crops | 17.5% | 14.1% | **9.9%** |

Method, more tests and the training recipe are in the
[model card](https://github.com/RivoksLab/tesseract-heb-rivok#readme). The 403 crops come
from private documents; GlotOCR-bench and HaZefira are public, and the repository's `eval/`
scripts rerun those numbers.

## Files

| file | SHA-256 |
|---|---|
| `heb_rivok.traineddata` (5.7 MB) | `dd9ba506b4ec6e62119a1334949824137692eada686687e51ac10dbb2f7a3df8` |
| `heb_rivok_lite.traineddata` (3.4 MB) | `a92584816adf5c9ae38afc255e3530477516318c21c2e711fd57b05535239360` |

## Licence

Apache-2.0, like the tessdata_best models and langdata_lstm text they are built on.
