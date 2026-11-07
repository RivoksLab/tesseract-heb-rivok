# Rerunning the public benchmarks

These scripts reproduce the "Public benchmarks" table of the model card for the int8 files in
`../tessdata/`. The data is not in this repository: each script downloads it at run time.

| benchmark | what is read | mode | score |
|---|---|---|---|
| [GlotOCR-bench](https://huggingface.co/datasets/cis-lmu/GlotOCR-bench), Hebrew | 100 lines, two image variants: `img_plain` and `img_old_document` | `--psm 6` | mean CER, the benchmark's own metric |
| [HaZefira](https://github.com/omilab/historical_press) lines | 4,013 line crops from 34 real 1874/1887 newspaper scans | `--psm 13` | CER: summed edits / summed reference characters |
| HaZefira pages | the same 34 pages, untranscribed areas painted white | `--psm 3` | word F1, order-free |

## Data terms

- **GlotOCR-bench** is published under an evaluation-only licence (no training). The dataset is
  gated on Hugging Face: accept its terms there and log in (`hf auth login`, or set `HF_TOKEN`)
  before the first run. None of it was used to train heb_rivok.
- **HaZefira** (OmiLab, Open University of Israel) states **no licence**. Use it for evaluation
  only, and never commit or redistribute it, or anything derived from it (crops, page images,
  transcriptions, predictions). `hazefira.py build` downloads it into a folder you choose;
  `eval/data/`, `eval/results/` and `eval/models/` are git-ignored for this reason.

## Requirements

- Tesseract 5 (the numbers below come from 5.5.0). One process per image, one thread each.
- Python 3.10+ with `pyarrow`, `huggingface_hub`, `Pillow` and `python-bidi`
  (`pip install pyarrow huggingface_hub pillow python-bidi`). Our runs used Python 3.12,
  pyarrow 26.0.1, huggingface_hub 2.4.0, Pillow 13.0.0, python-bidi 0.6.12.
- For the stock comparison: Ubuntu's `heb` and `eng` (`apt install tesseract-ocr-heb
  tesseract-ocr-eng`; these are tessdata_fast `heb` and `eng`), or `bash fetch_models.sh` and
  `--tessdata models/stock` (env `STOCK=models/stock` for `run_all.sh`).

## Run

Everything at once (about 20 minutes on 2 cores; predictions and scores land in `results/`):

```bash
bash run_all.sh
```

Or one at a time:

```bash
python -I glotocr.py --tessdata ../tessdata --lang heb_rivok
python -I glotocr.py --lang heb                                   # Ubuntu's heb

python -I hazefira.py build --data-dir data/hazefira              # once: download (44 MB, pinned commit, sha256 checked) and build
python -I hazefira.py lines --data-dir data/hazefira --tessdata ../tessdata --lang heb_rivok
python -I hazefira.py lines --data-dir data/hazefira --lang heb
python -I hazefira.py pages --data-dir data/hazefira --tessdata ../tessdata --lang heb_rivok
python -I hazefira.py pages --data-dir data/hazefira --lang heb+eng
```

Options: `--jobs N` (images in parallel, default 2), `--out FILE.jsonl` (keep the predictions;
a rerun of `hazefira.py` reuses them), `--data-dir` for a local copy of either dataset
(GlotOCR: a folder holding `Hebr_000.parquet`), `--psm` to try other modes.

## Expected results (int8 files, Tesseract 5.5.0)

| test | Ubuntu `heb` | heb_rivok_lite | **heb_rivok** |
|---|---:|---:|---:|
| GlotOCR-bench, `img_plain`: mean CER | 21.2% | 20.6% | **18.5%** |
| GlotOCR-bench, `img_old_document`: mean CER | 34.0% | 32.3% | **30.1%** |
| HaZefira, 4,013 lines, psm 13: CER | 17.45% | 14.06% | **9.93%** |
| HaZefira, 34 pages, psm 3: word F1 | 67.6% (`heb+eng`) | | **81.3%** |
| HaZefira, 34 pages, psm 3: word error rate in reading order | 47.9% (`heb+eng`) | 37.3% (`+eng`) | 88.5% |
| the same, `-l heb+heb_rivok` | | | **37.2%** (F1 78.9%) |

How each model is run: `heb_rivok` alone (it reads English itself); `heb_rivok_lite` and Ubuntu
`heb` without `+eng` on GlotOCR and the HaZefira lines; Ubuntu `heb+eng` on the pages, as an
OCRmyPDF user would run it. The word error rate in reading order counts words out of place as
well as wrong, so it shows that `heb_rivok` reads multi-column pages in the wrong column order
(model card, Limits); word F1 ignores order. The "best stock" column of the model card picks, per row, the best
of Google's four Hebrew models (tessdata_fast and tessdata_best `heb`, `script/Hebrew` fast and
best); run them the same way with `--tessdata` pointing at a folder that holds the model.

Rerunning these scripts on the machine behind the numbers gave every line and page the same
text as the original runs. Elsewhere, another Tesseract version, another CPU (Tesseract picks
AVX2, SSE or NEON code at run time) or another python-bidi (used to compare line texts in
display order) can move the last digit.

## Scoring details (`metrics.py`)

- **GlotOCR**: the benchmark's metric, copied from `cisnlp/GlotOCR-bench`
  (`evaluation/metrics/utils.py`): Levenshtein distance with whitespace removed, the minimum
  over the reference, its reverse, lower case and accents stripped, divided by the reference
  length and capped at 1, then averaged over the 100 lines. Each image is converted to RGB PNG
  and piped to `tesseract stdin stdout --psm 6`. The "no marks" figure also strips niqqud.
- **HaZefira lines**: both texts lose niqqud, cantillation and BiDi control characters and are
  NFC-normalised with single spaces; both are put in display order with python-bidi (base
  direction L and R) and the smaller edit distance counts. CER = summed edits / summed
  reference characters over all 4,013 lines. "Folded" also counts ׳ ״ ־ – as ' " - -.
- **HaZefira pages**: the page reference is OmiLab's line transcription, regions in the PAGE
  reading order. Both texts are cut into words (marks removed, ׳ ״ ־ – — folded, punctuation
  stripped from word edges, Hebrew letters and digits split apart) and compared as multisets:
  recall, precision, F1, and recall of words that contain digits.
- **HaZefira build**: each line crop is the bounding box of its TextLine polygon plus 6 px,
  from the unmasked scan; slanted lines can include bits of their neighbours, which is why psm
  13 (no layout analysis) is used: in psm 7 Tesseract's line finder drops about a third of the
  crops before recognition.
