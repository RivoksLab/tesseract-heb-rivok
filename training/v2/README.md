# How heb_rivok 2.0 and heb_rivok_lite 2.0 were trained, and how to do it again

Everything that went into the 2.0 models is public: the documents and source files are listed here,
and the lines, training lists, start models and snapshots are in the line dataset
[Rivok/tesseract-heb-rivok-v2-lines](https://huggingface.co/datasets/Rivok/tesseract-heb-rivok-v2-lines)
(CC BY-SA 4.0). The code in this folder rebuilds each step. The 1.0 inputs (Tesseract's langdata
training text and word lists, and the fonts of 1.0) are listed in `../RECIPE.md` and fetched by
`../fetch_inputs.sh` and `../fetch_fonts.sh`.

```bash
# the released files, from our snapshots (minutes)
WORK=work DATASET=dataset bash train_v2.sh ours release
# the whole training, from the 1.0 models (about 7 hours on 4 CPU cores); see train_v2.sh for the inputs
WORK=work DATASET=dataset bash train_v2.sh
# the source documents, from where they are published (~700 MB; --with-dicta adds Dicta's model, 1.2 GB)
python3 fetch_v2_sources.py --out v2_sources
```

## What was checked

On 2026-11-23, with the published inputs and the code in this folder:

- **The released files:** `train_v2.sh ours release` gives all four (`tessdata_best/` and `tessdata/`
  of both models) byte for byte.
- **The real lines:** `lines/build_lines.py`, run on the 14 scans from Wikimedia Commons (each matching
  its sha1 in `books.tsv`), gives the 24,643 lines of the training pool and the 307 held-out Seder Tefilot
  lines: the same lines, in the same order, with the same labels, and crops identical pixel for pixel.
  Re-running Tesseract 5.5.0 with heb_rivok 1.0.1 on 24 sampled pages gave our hOCR exactly.
- **Everything before training:** `train_v2.sh` run up to training (with our 60,000-iteration checkpoint
  in place of the first stage) gives all 66,000 synthetic lines (images, labels, manifest), all 90,643
  `.box` and `.lstmf` files, the character sets and model files the runs start from (dictionaries and
  recoder included), and all four training lists with their eval lists, byte for byte.
- **Dev scores:** `train/dev_score.py` gives our stored scores exactly on the public dev sets. One
  exception: `heb_rivok`'s first stage was scored before the scorer started counting holam haser for
  vav (U+05BA) as holam, which it did from the second stage on. Three Tanakh dev lines carry that mark,
  so the first stage's logged Tanakh scores are 0.01-0.02 points higher than this scorer gives.
- **What is not bit-exact:** `lstmtraining` itself. Retraining on another machine or with another number
  of threads gives slightly different weights, which is why the dataset also has our snapshots.

## Files

| File | What it holds |
|---|---|
| `books.tsv` | The 14 scanned books whose printed lines were trained on: Wikimedia Commons file, sha1, licence tag, Hebrew Wikisource index, and how many lines each model used |
| `pages.tsv` | Their 545 pages: the 522 training pages and the 23 held-out Seder Tefilot pages, each with the Hebrew Wikisource revision whose transcription labelled the lines |
| `text_sources.tsv` | Vowelled text and model files, with URL, sha256 and licence: Tanakh (Miqra according to the Masorah), Mishnah (Torat Emet), the Project Ben-Yehuda public-domain dump, Dicta's vowelling model |
| `wikipedia_articles.tsv` | The 271 Hebrew Wikipedia articles (with revision) whose first 4,000 paragraphs Dicta vowelled for the "modern vowelled" lines |
| `benyehuda_works.tsv` | The Project Ben-Yehuda works in the vowelled pool (those with dense niqqud) |
| `fonts.tsv` | The 165 font files used to render synthetic lines, with sha256, licence and source |
| `label_changes.tsv` | The 349 real lines whose training label differs from the Wikisource text, with both texts and the reason |
| `fetch_v2_sources.py` | Downloads the books, pages, text sources, Wikipedia revisions and fonts, and checks each against its hash |
| `train_v2.sh` | The whole recipe, step by step, with the exact seeds and settings |
| `lines/` | `build_lines.py`: cuts and labels the real lines (page images, Tesseract hOCR, alignment to the Wikisource text, the label rules in `labels.py`) |
| `synth/` | `gen_lines_v2.py` (the synthetic-line generator, built on `../gen_lines.py`), the font order and capabilities it used, and `collect_fonts.py` |
| `train/` | Training lists (`build_lists.py`, `stage_lists.py`), the chunked training driver (`train_v2.py`), dev scoring (`dev_score.py`) and averaging (`average.py`, with `tesslstm.py`, which reads and writes the network inside a `.traineddata`) |

## What each model was trained on

Both models continued training from their 1.0 versions with Tesseract's `lstmtraining` (fine-tuning
only, learning rate 1e-4). They started from the average of 1.0's last five training states, not from
the released 1.0 files (`../RECIPE.md` has the 1.0 runs); the dataset has both start models.

**Real printed lines.** Lines cut from the 14 books in `books.tsv` (public-domain books printed
between 1876 and 1926, plus one 2025 book its author released into the public domain), labelled with
the proofread Hebrew Wikisource transcription of each page. `heb_rivok` saw 15,663 distinct lines,
`heb_rivok_lite` 14,291, of a pool of 24,643. Three volumes of Hyman's *Toldoth Tannaim veAmoraim* (1910)
are most of them, and the vowelled 1885 Seder Tefilot is the next largest. Lines whose transcription
differs from the print (modernised spelling, expanded abbreviations, editor's notes) were dropped, and so
were all lines on pages that Wikisource marks as validated (quality 4), because another benchmark draws
its test pages from those.

**Changes to the Wikisource text.** We did not edit labels by hand. 349 of the 20,248 real lines that
either model used have an automatic change, listed line by line in `label_changes.tsv`:

| Change | Lines |
|---|---:|
| A holam typed where the print has a sin dot, corrected (Seder Tefilot) | 141 |
| Alternative ayin (U+FB20) written as ע (Hyman) | 122 |
| Sof pasuq ׃ written as a colon, as in the synthetic lines | 31 |
| Zero-width joiner removed | 24 |
| Vowels removed where the OCR saw none | 24 |
| A word the print breaks at the line end, which Wikisource joins, split back across the two lines | 10 |

The vowel rule was meant for vowels that Wikisource adds and the print lacks. It was right for the 16
Mishneh Torah lines, but 8 of the lines (7 in the Seder Tefilot and one title) are vowelled in print, so
their labels wrongly lack the vowels. They are marked in the file.

**Synthetic lines**, rendered with the fonts in `fonts.tsv` and damaged to look scanned, faxed, faded
or photographed:

| Stream | Text |
|---|---|
| Replay of 1.0's mix | Tesseract's langdata Hebrew and English training text and word lists (as in 1.0) |
| Vowelled (also in prayer-book typefaces) | Tanakh (all books but Ruth, Lamentations, Esther and Song of Songs, kept out for development), Mishnah, Ben-Yehuda works, modern Wikipedia paragraphs vowelled by Dicta |
| Look-alike letters | Pairs of real words that differ only in letters that are easy to confuse (ד/ר, ב/כ, ז/ו, ...), from the langdata word list |
| Fonts | Plain Hebrew from the langdata training text, in every font |
| Hard negatives | Words full of thin letters that look like digits or each other (ו ן י ס ם ...), from the langdata word list; specks, column rules and slivers of neighbouring columns beside lines, labelled as nothing |
| Signs, document numbers | Short words, prices, dates and IDs, generated |
| Latin | Lines without Hebrew (English and numbers), in `heb_rivok` only |
| Rare marks | Lines from the vowelled sources that carry the rarest marks (hataf qamats, qubuts, meteg, ...) |

**The runs** (`train_v2.sh` has every command):

- `heb_rivok`, first stage: 60,000 iterations, 25% real lines. The first 8,000 iterations ran on an
  80,932-line list; then the 2,000 Latin lines were added (82,932 lines).
- `heb_rivok`, second stage: 20,000 iterations from the first stage's last checkpoint, on a
  20,000-line list (35% real, 3% hataf qamats lines, 2.5% Latin), each line once.
- `heb_rivok_lite`: 60,000 iterations on 60,000 lines (35% real), without the lines that contain Latin
  letters (the model has none).
- **The released files** average the weights of the last 8 snapshots of `heb_rivok`'s second stage
  (66,000-80,000) and the last 6 of `heb_rivok_lite` (50,000-60,000). For `heb_rivok`, 23 unused Hebrew
  characters are then added to the character set (`../pad_rtl.py`) so that Tesseract lays its pages out
  right to left.

## Choosing checkpoints

These sets were used only to choose checkpoints, never to train, and none of the model card's test
sets was used for either:

- 1,500 lines of *Milhamot Hashem* (1923) and 307 lines from 23 held-out pages of the Seder Tefilot,
  both from Hebrew Wikisource;
- 400 synthetic lines from the four held-out Tanakh books;
- 500 synthetic lines of 1.0's mix, as a check against forgetting (reported, not in the score);
- 253 lines from private receipts (not published).

All are in the dataset except the receipts. The dev score was the mean error of the first four sets
above and the receipts. Without the receipts (`train/dev_score.py` as published), the choices come out
as follows:

- **`heb_rivok_lite`:** the same snapshots (50,000-60,000).
- **`heb_rivok`:** the best window would be the last 4 snapshots (74,000-80,000) instead of the last 8.
  The gap in the dev score is 0.02 points (4.386 against 4.404), within the noise between neighbouring
  snapshots. The 2.0 files are not changed for it.
- **When to stop:** for `heb_rivok`'s first stage and for `heb_rivok_lite`, the stop rule would not have
  fired either way. For `heb_rivok`'s second stage, it would fire at 78,000 instead of 72,000; that stage
  was run to 80,000 regardless, so that its list was used exactly once.

## Dictionaries

Both models keep Tesseract's word, punctuation and number lists. Since 2.0 has more characters, they
were rebuilt from the same source lists as 1.0 (`../RECIPE.md`):

- **`heb_rivok`:** langdata_lstm's Hebrew word list plus the first 150,000 entries of its English word
  list that are made of ASCII letters (`hebeng.wordlist`), and the Hebrew and English punctuation and
  number patterns. 2.0's list has 262 more words than 1.0's, all of them words with one of the six
  added marks, which 1.0 could not store.
- **`heb_rivok_lite`:** langdata_lstm's Hebrew word list, punctuation and number patterns. 2.0 stores
  4,570 more of the list's words, those with the marks it added.

## Licences

- **Scans:** public domain, per each file's licence tag on Wikimedia Commons (`books.tsv`).
- **Transcriptions:** Hebrew Wikisource, CC BY-SA 4.0. The Wikisource text quoted in `label_changes.tsv`
  is under the same licence, and so is the line dataset.
- **Text sources:** Tanakh (MAM), CC BY-SA 4.0; Mishnah (Torat Emet) and Project Ben-Yehuda, public
  domain; Hebrew Wikipedia, CC BY-SA 4.0; Dicta's model, CC BY 4.0.
- **Fonts:** OFL-1.1, GPL with font exception, and the DejaVu licence (see `fonts.tsv`). Three Noto
  files whose download we did not record are in the dataset, under OFL-1.1.
- **Code in this folder:** Apache-2.0, as the repository.

Credits are in `../../NOTICE`.
