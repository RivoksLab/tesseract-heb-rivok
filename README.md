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
  - niqqud
---

# heb_rivok: better Hebrew models for Tesseract

**[Article with examples](https://rivoklabs.com/research/tesseract-hebrew)** · [GitHub release](https://github.com/RivoksLab/tesseract-heb-rivok/releases/tag/v2.0) · [Hugging Face](https://huggingface.co/Rivok/tesseract-heb-rivok)

Two drop-in Tesseract 5 models for printed Hebrew, fine-tuned by
[Rivok Labs](https://rivoklabs.com) from Google's own Hebrew models:

- **`heb_rivok`**: Hebrew, English and digits in one model, with niqqud. Fine-tuned from Google's
  `script/Hebrew` (tessdata_best). The more accurate of the two; start here.
- **`heb_rivok_lite`**: Hebrew and digits, with niqqud, fine-tuned from Google's standard `heb`
  (tessdata_best). About as fast as the standard Hebrew model; use it with `+eng` for English.

Using `heb_rivok` in a project or product? We'd love to hear about it: [ronen@rivoklabs.com](mailto:ronen@rivoklabs.com).

**Version 2.0** reads old print and vowelled text (niqqud) far better than 1.0, and modern documents as
well as before. On 1,087 lines from four Hebrew books printed between 1872 and 1922, Ubuntu's standard
Hebrew model gets **11.3%** of characters wrong, 1.0 **8.1%** and 2.0 **2.7%**. On a vowelled 1885
prayer book it keeps **82%** of the vowel marks, against 23% for Ubuntu's model. See [Changes](#changes).

Each model is one file: no new software, and it works wherever Tesseract does, OCRmyPDF and
Paperless-ngx included.

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

Or with pip: `pip install tesseract-heb-rivok`, then
`tesseract page.png out --tessdata-dir "$(python -m tesseract_heb_rivok)" -l heb_rivok`.

- **No `+eng` with `heb_rivok`.** English is already in the model.
- **Line crops:** use `--psm 13` for crops that hold one line, not `--psm 7` (see Limits).

| file | size | use | SHA-256 |
|---|---:|---|---|
| `tessdata/heb_rivok.traineddata` | 5.7 MB | recognition (int8) | `09ef1830f3393c590791abf9c18deca14450c8e5d3b556249d63b0183cf7a426` |
| `tessdata/heb_rivok_lite.traineddata` | 3.5 MB | recognition (int8), with `+eng` | `59da073ad0bca9449fa5a965996e49d0e2bee3c17e8b0b4e2769d67a7ce34c40` |
| `tessdata_best/heb_rivok.traineddata` | 12.4 MB | the most accurate (float); further fine-tuning | `673a4193be1f77565942a69480f3c28c7de9633f25f1fa318162f18fa258ba1b` |
| `tessdata_best/heb_rivok_lite.traineddata` | 6.1 MB | further fine-tuning (float) | `3e70e4bdaac955add0cf25f4666ece244795cf7d523f4a4d3dfe52fa1b8b13d5` |

The float `heb_rivok` reads vowels a little better than the int8 file (below) at about 1.3× its time
per line. Use either with `-l heb_rivok`; put the one you want in your tessdata folder.

## Results

Character error rate (CER: the share of characters read wrong, missed or added; lower is
better) unless marked. "Words found" is word F1 against the page's own text; higher is better.

Almost everyone running Tesseract on Hebrew uses Google's small "fast" `heb`, the file Ubuntu,
Debian and Homebrew install. `script/Hebrew` (tessdata_best) is Google's biggest Hebrew model and the
one `heb_rivok` was fine-tuned from. Our columns are the downloadable int8 files.

| test | Ubuntu `heb` | Google `script/Hebrew` | heb_rivok 1.0 | **heb_rivok 2.0** | heb_rivok_lite 2.0 |
|---|---:|---:|---:|---:|---:|
| 150 lines from real documents, Hebrew only | 4.7% | 4.0% | 1.4% | **1.2%** | 1.7% |
| 253 lines from real documents, with numbers | 4.5% | 3.5% | 1.6% | **1.6%** | 1.5% |
| Four old books (1872–1922): 1,087 lines | 11.3% | 7.8% | 8.1% | **2.7%** | 4.8% |
| … the vowelled Selichot 1885: vowel marks read (higher is better) | 23% | 56% | 50% | **82%** | 61% |
| [GlotOCR-bench](https://huggingface.co/datasets/cis-lmu/GlotOCR-bench) Hebrew, clean images (many with niqqud) | 21.2% | 20.7% | 18.5% | **10.9%** | 13.2% |
| GlotOCR-bench Hebrew, "old document" images | 34.0% | 31.8% | 30.1% | **25.0%** | 27.8% |
| [HaZefira](https://github.com/omilab/historical_press) newspaper, 1874/1887: 4,013 lines | 17.5% | 14.3% | 9.9% | **4.8%** | 6.3% |
| HaZefira: 34 pages, words found (higher is better) | 67.6% | 75.5% | 81.3% | **92.1%** | 91.0% |
| 209 lines mixing Hebrew and English | 19.3% | 16.4% | 14.0% | **14.3%** | 15.1% |

Ubuntu `heb` and the lite model run with `+eng` on pages and mixed lines. On unvowelled old print the
lite model is close to `heb_rivok`; on vowelled text `heb_rivok` stays well ahead.

### Vowels (niqqud) in detail

On 862 vowelled lines (a 19th-century prayer book, the 1885 Selichot and Tanakh text), share of the
vowel marks each model reads (and, of the marks it outputs, the share that are right):

| model | marks read | marks right | words exact, vowels included |
|---|---:|---:|---:|
| Ubuntu `heb` | 30% | 56% | 5% |
| Google `script/Hebrew` (best) | 62% | 74% | 22% |
| heb_rivok 1.0 | 60% | 77% | 29% |
| **heb_rivok_lite 2.0** | **73%** | **86%** | **49%** |
| **heb_rivok 2.0** | **85%** | **92%** | **66%** |
| heb_rivok 2.0, float | 88% | 93% | 69% |

Reliable (85% or more read): shva, qamats, patah, hiriq, holam, segol, dagesh, hataf patah, shin
and sin dots. Tsere and hataf segol are read most of the time. Weak: qamats qatan (in most fonts the
same shape as qamats), qubuts, meteg and hataf qamats.

Reading the vowels off the page also beats reading the letters and adding vowels with a vowelling
model: given the correct letters, Dicta's vowelling model (dictabert-large-char-menaked) gets 53% of
these words exactly right, against 66% for `heb_rivok` 2.0. Old prints and prayer books often vowel
differently from modern Hebrew.

### Settings and data

- **Settings:** line tests use `--psm 13`; pages use Tesseract's default; GlotOCR uses `--psm 6`, as its authors do. `heb` (and `heb_rivok_lite`) run with `+eng` on pages and mixed lines.
- **Public, rerunnable:** GlotOCR-bench (LMU Munich; evaluation-only licence) and HaZefira (OmiLab, Open University of Israel; real scans with transcriptions). Scripts are in `eval/`; they download the data, which is not redistributed here.
- **Four old books:** pages of four public-domain books (Vilna 1872, 1883, 1885; Warsaw 1922) scanned on Wikimedia Commons, with the validated Hebrew Wikisource transcriptions (CC BY-SA 4.0) split into printed lines and checked by hand. No page of these books was used in training. Ask us if you want the set.
- **Ours, private:** the 150 + 253 + 209 lines are cut from real documents whose owners did not agree to publication.
- **No overlap:** none of the test images were used in training.

## Examples

See them line by line, with errors marked, in the
[article](https://rivoklabs.com/research/tesseract-hebrew#examples); `examples/` regenerates them:
whole pages read by Ubuntu's `heb` and by `heb_rivok`, default settings.

| page | Ubuntu `heb` | heb_rivok 1.0 | heb_rivok 2.0 |
|---|---:|---:|---:|
| A novel, printed in Vilna in 1872 (real scan) | 10.5% | 4.4% | **1.2%** |
| A travel book, printed in Warsaw in 1922 (real scan) | 6.2% | 1.3% | **1.1%** |
| A bank letter, faxed (text we wrote) | 6.7% | 1.4% | **0.3%** |
| A rental agreement, photocopied (same) | 4.3% | 0.8% | 1.3% |

- **We chose these pages** to show what the model fixes. Across all seven documents we wrote, the rates were 4.5% (Ubuntu `heb`), 1.0% (1.0) and 1.1% (2.0).
- **Fonts:** the documents we wrote use fonts that 1.0 never saw in training. 2.0's training fonts include them (with other texts), so for 2.0 they test unseen text in known fonts.
- **Scoring:** typographic variants (״ for ") and spaces before punctuation are not counted.
- **Real scans:** public domain via Wikimedia Commons (National Library of Israel scans). Their reference text comes from Hebrew Wikisource (validated pages, CC BY-SA 4.0).

## Limits

- **Pages with more than one column** come out right column first since 1.0.1.
  - Tesseract decides whether a model reads right to left by counting its characters (`UNICHARSET::major_right_to_left`). `heb_rivok` knows the 52 English letters, so its character list is padded with 23 Hebrew characters it never outputs (presentation forms; `training/pad_rtl.py`). The lite model has no English letters and needs no padding.
  - Upstream, tesseract-ocr/tesseract#4629 proposes a setting that would make the padding unnecessary.
- **Short lines that mix Hebrew and English.**
  - When a line is read on its own (`--psm 7`/`13`), or forms its own paragraph on a page (a heading, a table cell), Tesseract picks its direction from its words, and a tie goes to left-to-right. `הרצאה על Machine Learning` then comes out as `Machine Learning הרצאה על`.
  - This is Tesseract's code (`ResultIterator::CurrentParagraphIsLtr`), not the model, and stock models do the same. Reported upstream as [tesseract-ocr/tesseract#4680](https://github.com/tesseract-ocr/tesseract/issues/4680); `examples/direction/` reproduces it.
  - On the 209 real mixed lines, nearly all of `heb_rivok` 2.0's 14.3% is word order: counting characters regardless of order, 1.7%.
  - Long paragraphs are not affected.
- **Many mixed lines? Try [paddleocr-hebrew](https://github.com/RivoksLab/paddleocr-hebrew).**
  - It is our deep-learning Hebrew OCR. It assembles each line in Hebrew reading order: 3.6% on the same 209 lines.
  - It is much slower on a CPU (about 55 s per page) and fast on an NVIDIA GPU. On old print `heb_rivok` 2.0 is now ahead (HaZefira lines: 4.8% against 9.9%).
- **`--psm 7` drops some line crops.** Tesseract's line finder throws away crops with an underline or with pieces of the neighbouring lines before the model sees them. `--psm 13` reads them.
- **Niqqud:** see Vowels above. Holam haser for vav (U+05BA) comes out as holam (U+05B9); cantillation marks (טעמים) are not read.
- **Stray vowel marks on text without niqqud:** about one mark every 100 words for `heb_rivok` (the same as Google's `script/Hebrew`) and one every 240 for the lite model (Ubuntu's `heb`: one every 320).
- **Not trained for:** Rashi script, handwriting, decorative display fonts.
- **Speed** (Intel Core i7-10700T desktop, one thread, the model loaded once):
  - `heb_rivok`: 3.5 s per newspaper page (HaZefira) against 2.4 s for Ubuntu's `heb` alone and 4.5 s for `heb+eng`, which Hebrew pages with English need; 0.07 s per line against 0.06 s. The float file: 5.9 s per page, 0.09 s per line.
  - `heb_rivok_lite`: the same as Ubuntu's `heb` (2.4 s per newspaper page, 0.06 s per line); with `+eng` 4.0 s.
  - Loading `heb_rivok`'s larger file takes about 0.1 s more each time Tesseract starts, so on many small images one process per image is about 3× slower than Ubuntu's `heb`.

## How they were trained

- **Fine-tuning only.** Tesseract's `lstmtraining` continued training the 1.0 models (themselves fine-tuned from Google's) at learning rate 1e-4, on a CPU: about 4.5 hours for `heb_rivok` and 2 hours for the lite model. The released files are averages of the weights of the last training checkpoints, chosen on held-out development pages.
- **Real lines (new in 2.0):** about 24,600 printed lines from public-domain Hebrew books of the 19th and early 20th century (scans on Wikimedia Commons, validated Hebrew Wikisource transcriptions), cleaned of modernised spellings. The test books were kept out.
- **Synthetic lines:** modern documents, Hebrew with English (for `heb_rivok`), vowelled text (Tanakh, Mishnah, Project Ben-Yehuda, and modern text vowelled with Dicta's model), look-alike letters, signs and document numbers, rendered in about 165 open fonts and damaged to look scanned, faxed, faded or photographed.
- **Character list:** 2.0 adds hataf vowels, qubuts, qamats qatan, meteg and rafe; the lite model also gains tsere, holam and the shin/sin dots it lacked.
- **Want the details?** This is a summary; the full 1.0 recipe is in `training/`. For more on 2.0's data and training, write to us: ronen@rivoklabs.com.

## Changes

- **2.0** (2026-11-13): both models retrained with real old-print lines and much more vowelled text. Old print and niqqud are far better (see Results); modern documents are about the same. New characters: hataf vowels, qubuts, qamats qatan, meteg, rafe (and, in the lite model, tsere, holam and the shin/sin dots). The float `heb_rivok` is now offered for recognition too. Known regression: two of the seven example documents lose a few characters (Examples).
- **1.0.1** (2026-11-07): `heb_rivok` lays pages out right to left, so pages with columns come out in the right order (see Limits). Only its character list changed.
- **1.0** (2026-11-07): first release.

## Licence and credits

- **Licence:** Apache-2.0, the same as the models they are built on.
- **Based on:** Google's tessdata_best `script/Hebrew` and `heb`, and the langdata_lstm training text, all Apache-2.0 (see `NOTICE`).
- **Training text and images (2.0):** Hebrew Wikisource transcriptions (CC BY-SA 4.0) of public-domain scans on Wikimedia Commons; Miqra according to the Masorah (Hebrew Wikisource, CC BY-SA 4.0); Mishnah (Torat Emet, public domain); Project Ben-Yehuda (public domain); dicta-il/dictabert-large-char-menaked (CC BY 4.0), used to vowel modern text.
- **Fonts:** used only to render training images (Google Fonts and SIL, OFL-1.1; Culmus, GPL-2.0 with font exception); no font is distributed here.
- **Made by:** Rivok Labs. Contact: ronen@rivoklabs.com
- **Credit:** If you use it, a credit or a link back to this repository is appreciated.
