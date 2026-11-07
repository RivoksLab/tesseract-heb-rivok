---
license: apache-2.0
language:
  - he
  - en
tags:
  - ocr
  - tesseract
  - hebrew
  - traineddata
  - printed-text
---

# heb_rivok: better Hebrew models for Tesseract

**[Article with examples](https://rivoklabs.com/research/tesseract-hebrew)** · [GitHub release](https://github.com/RivoksLab/tesseract-heb-rivok/releases/tag/v1.0) · [Hugging Face](https://huggingface.co/Rivok/tesseract-heb-rivok)

Two drop-in Tesseract 5 models for printed Hebrew, fine-tuned by
[Rivok Labs](https://rivoklabs.com) from Google's own Hebrew models:

- **`heb_rivok`**: Hebrew, English and digits in one model. Fine-tuned from Google's
  `script/Hebrew` (tessdata_best). The more accurate of the two; start here.
- **`heb_rivok_lite`**: Hebrew and digits, fine-tuned from Google's standard `heb`
  (tessdata_best). Smaller and about three times faster per line; use it with `+eng`.

On 403 lines cut from real Hebrew documents, Ubuntu's standard Hebrew model gets **4.7%** of
characters wrong and `heb_rivok` **1.4%**. Each model also beats the Google model it was
fine-tuned from on every test we ran, including two public benchmarks. Each is one file: no new
software, and it works wherever Tesseract does, OCRmyPDF and Paperless-ngx included.

## Quick start

```bash
# check the hash first (table below), then copy the file next to Tesseract's other models
# (`tesseract --list-langs` prints the folder)
sha256sum heb_rivok.traineddata
sudo cp heb_rivok.traineddata /usr/share/tesseract-ocr/5/tessdata/

tesseract page.png out -l heb_rivok
ocrmypdf -l heb_rivok scanned.pdf searchable.pdf

tesseract page.png out -l heb_rivok_lite+eng       # the lite model needs eng for English
```

- **No `+eng` with `heb_rivok`.** English is already in the model.
- **Line crops:** use `--psm 13` for crops that hold one line, not `--psm 7` (see Limits).

| file | size | use | SHA-256 |
|---|---:|---|---|
| `tessdata/heb_rivok.traineddata` | 5.7 MB | recognition (int8) | `ae48e574583afa82832704d402a17cc496ecfd9d58ef29cc92a124de57d9bda8` |
| `tessdata/heb_rivok_lite.traineddata` | 3.4 MB | recognition (int8), with `+eng` | `a92584816adf5c9ae38afc255e3530477516318c21c2e711fd57b05535239360` |
| `tessdata_best/heb_rivok.traineddata` | 12.3 MB | further fine-tuning (float) | `706ad13c68235c1832b7d19ac3925ee3fc105eba2da83181a0f810b9c39c35ce` |
| `tessdata_best/heb_rivok_lite.traineddata` | 6.0 MB | further fine-tuning (float) | `baceb6c52bf69550ffae52acad9297619a1b7fc8553890caf299798f7348e4cd` |

## Results

Character error rate (CER: the share of characters read wrong, missed or added; lower is
better) unless marked. "Words found" is word F1 against the page's own text; higher is better.

### The file you have against the file you can download

Almost everyone running Tesseract on Hebrew uses Google's small "fast" `heb`, the file Ubuntu,
Debian and Homebrew install. Scores below are for the downloadable int8 `heb_rivok`.

| test | Ubuntu `heb` | **heb_rivok** |
|---|---:|---:|
| 403 lines from real documents, Hebrew only | 4.7% | **1.4%** |
| the same, lines with numbers | 4.5% | **1.6%** |
| [GlotOCR-bench](https://huggingface.co/datasets/cis-lmu/GlotOCR-bench) Hebrew, clean images (many with niqqud) | 21.2% | **18.5%** |
| GlotOCR-bench Hebrew, "old document" images | 34.0% | **30.1%** |
| [HaZefira](https://github.com/omilab/historical_press) newspaper, 1874/1887: 4,013 lines | 17.5% | **9.9%** |
| HaZefira: 34 pages, words found | 67.6% | **81.3%** |
| 209 lines mixing Hebrew and English | 19.3% | **14.0%** |
| our 7 example documents (see below) | 4.5% | **1.0%** |

### Same network, before and after our fine-tuning

`heb_rivok` starts from Google's biggest Hebrew model, `script/Hebrew`, so part of the gain over
Ubuntu's `heb` could be the bigger model. This table isolates the fine-tuning: the same network
with the same starting weights, both in full precision.

| test | Google `script/Hebrew` (best) | **heb_rivok (best)** |
|---|---:|---:|
| 403 lines from real documents, Hebrew only | 4.0% | **1.4%** |
| the same, lines with numbers | 3.5% | **1.6%** |
| GlotOCR-bench Hebrew, clean images | 20.7% | **18.0%** |
| GlotOCR-bench Hebrew, "old document" images | 31.8% | **29.5%** |
| HaZefira: 4,013 lines | 14.3% | **9.9%** |
| HaZefira: 34 pages, words found | 75.5% | **81.2%** |
| 209 lines mixing Hebrew and English | 16.4% | **14.2%** |

### heb_rivok_lite

| test | Ubuntu `heb` | Google tessdata_best `heb` (its start) | **heb_rivok_lite** |
|---|---:|---:|---:|
| 403 lines from real documents, Hebrew only | 4.7% | 4.0% | **1.9%** |
| the same, lines with numbers | 4.5% | 4.3% | **1.6%** |
| GlotOCR-bench Hebrew, clean images | 21.2% | 24.6% | **20.6%** |
| GlotOCR-bench Hebrew, "old document" images | 34.0% | 33.5% | **32.3%** |
| HaZefira: 4,013 lines | 17.5% | 17.7% | **14.1%** |
| HaZefira: 34 pages, words found | 67.6% | 69.7% | **76.6%** |

### Settings and data

- **Settings:** line tests use `--psm 13`; pages use Tesseract's default; GlotOCR uses `--psm 6`, as its authors do. `heb` (and `heb_rivok_lite`) run with `+eng` on pages and mixed lines.
- **Public, rerunnable:** GlotOCR-bench (LMU Munich; evaluation-only licence) and HaZefira (OmiLab, Open University of Israel; real scans with transcriptions). Scripts are in `eval/`; they download the data, which is not redistributed here.
- **Ours, private:** the 403 lines are cut from real documents whose owners did not agree to publication.
- **No overlap:** none of the test images were used in training.

### Where the gains come from

On the 403 real lines, the errors stock Tesseract makes most often are almost gone:
- stray or missing full stops: 238 → 15;
- ם read as ס: 28 → 1;
- dropped digits: 85 → 5.

By document type, CER moves:
- medical 6.1% → 0.9%;
- news 3.9% → 0.7%;
- legal 1.9% → 0.4%;
- scanned forms 4.7% → 1.2%;
- old newspapers 16.3% → 8.5%.

Both models also output characters the stock Hebrew models lack: maqaf ־, geresh ׳ and
gershayim ״, plus en dash, bullet and €.

## Examples

See them line by line, with errors marked, in the
[article](https://rivoklabs.com/research/tesseract-hebrew#examples); `examples/` regenerates them:
whole pages read by Ubuntu's `heb` and by `heb_rivok`, default settings.

| page | Ubuntu `heb` | heb_rivok |
|---|---:|---:|
| A novel, printed in Vilna in 1872 (real scan) | 10.5% | 4.4% |
| A travel book, printed in Warsaw in 1922 (real scan) | 6.2% | 1.3% |
| A bank letter, faxed (text we wrote, font not used in training) | 6.7% | 1.4% |
| A rental agreement, photocopied (same) | 4.3% | 0.8% |

- **We chose these pages** to show what the model fixes. Across all seven documents we wrote, the rates were 4.5% and 1.0%.
- **Scoring:** typographic variants (״ for ") and spaces before punctuation are not counted.
- **Real scans:** public domain via Wikimedia Commons (National Library of Israel scans). Their reference text comes from Hebrew Wikisource (validated pages, CC BY-SA 4.0).

## Limits

- **Short lines that mix Hebrew and English.**
  - When a line is read on its own (`--psm 7`/`13`), or forms its own paragraph on a page (a heading, a table cell), Tesseract picks its direction from its words, and a tie goes to left-to-right. `הרצאה על Machine Learning` then comes out as `Machine Learning הרצאה על`.
  - This is Tesseract's code (`ResultIterator::CurrentParagraphIsLtr`), not the model, and stock models do the same. We are reporting it upstream.
  - On 209 real mixed lines (`--psm 7`), `heb_rivok` scores 14.2% as Tesseract orders the words, and 4.9% with the same words put back in right-to-left order.
  - Long paragraphs are not affected.
- **Many mixed lines? Try [paddleocr-hebrew](https://github.com/RivoksLab/paddleocr-hebrew).**
  - It is our deep-learning Hebrew OCR. It assembles each line in Hebrew reading order: 3.6% on the same 209 lines.
  - It is much slower on a CPU: about 55 s per page against 6.4 s for `heb_rivok` on a 4-core desktop. It is fast on an NVIDIA GPU.
  - On old print the two are level (HaZefira lines: 9.9% each).
- **`--psm 7` drops some line crops.** Tesseract's line finder throws away crops with an underline or with pieces of the neighbouring lines before the model sees them. `--psm 13` reads them.
- **Niqqud (vowel marks):** the 12 common marks are read; cantillation marks (טעמים) are not.
- **Not trained for:** Rashi script, handwriting, decorative display fonts.
- **Speed:**
  - `heb_rivok` uses Google's larger network: about 0.24 s per line against 0.07 s for Ubuntu's `heb`; on whole pages, 6.4 s against 5.2 s.
  - `heb_rivok_lite` runs at the standard model's speed.

## How they were trained

The full recipe and scripts are in `training/` (`training/RECIPE.md`).

- **Fine-tuning only.** Tesseract's `lstmtraining` continued training Google's model with `--old_traineddata`, at learning rate 1e-4, on a CPU.
  - `heb_rivok`: 60,000 lines, 20,000 iterations (about 2.5 hours on 4 cores).
  - `heb_rivok_lite`: 50,000 lines, 15,000 iterations.
- **No real documents in training.** Every training image is a synthetic text line:
  - Tesseract's own Hebrew training text ([langdata_lstm](https://github.com/tesseract-ocr/langdata_lstm), Apache-2.0);
  - document-style lines we generated: prices with ₪, dates, phone numbers, abbreviations such as ש"ח and עו"ד, prefixed numbers such as ב-2024, list items;
  - for `heb_rivok`, also Hebrew lines with English inside (terms in brackets, `ל-Facebook`, names, acronyms, URLs, citations) and English-only lines.
- **Rendering:** Pillow with raqm, in 50 open fonts (Google Fonts Hebrew families, DejaVu, GNU FreeFont, Liberation). The images were then damaged to look scanned: blur, low resolution, noise, JPEG, specks, binarisation, letterpress ink, slight rotation, underlines, slivers of neighbouring lines.
- **Labels:** visual order with an explicit line direction. Brackets are labelled the way Tesseract outputs them.
- **Dictionary:** Tesseract's Hebrew word list; for `heb_rivok`, plus the 150,000 most frequent English words (langdata_lstm).
- **History:**
  - round 1 added maqaf, geresh and gershayim;
  - round 2 fixed the bracket labels and added realistic damage;
  - round 3 restored lines with niqqud (→ `heb_rivok_lite`);
  - round 4 moved to the bigger `script/Hebrew` model with mixed Hebrew/English lines (→ `heb_rivok`).

## Licence and credits

- **Licence:** Apache-2.0, the same as the models they are built on.
- **Based on:** Google's tessdata_best `script/Hebrew` and `heb`, and the langdata_lstm training text, all Apache-2.0 (see `NOTICE`).
- **Fonts:** used only to render training images; no font is distributed here.
- **Made by:** Rivok Labs. Contact: ronen@rivoklabs.com
