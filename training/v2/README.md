# Training data of heb_rivok 2.0 and heb_rivok_lite 2.0

This folder lists every document and source file that went into the 2.0 models. Run
`fetch_v2_sources.py` to download all of them from where they are published. Nothing here is
redistributed: the files hold names, page numbers, revision IDs and hashes, and the script checks
each download against them. The 1.0 inputs (Tesseract's langdata training text and word lists, and the
fonts of 1.0) are listed in `../RECIPE.md` and fetched by `../fetch_inputs.sh` and `../fetch_fonts.sh`.

```bash
python3 fetch_v2_sources.py --out v2_sources          # ~700 MB; add --with-dicta for Dicta's model (1.2 GB)
python3 fetch_v2_sources.py --only books,pages        # just the scanned books and their transcriptions
```

## Files

| File | What it lists |
|---|---|
| `books.tsv` | The 14 scanned books whose printed lines were trained on: Wikimedia Commons file, sha1, licence tag, Hebrew Wikisource index, and how many lines each model used |
| `pages.tsv` | The 522 pages those lines come from: Hebrew Wikisource page title and the exact revision whose transcription labelled the lines |
| `text_sources.tsv` | Vowelled text and model files, with URL, sha256 and licence: Tanakh (Miqra according to the Masorah), Mishnah (Torat Emet), the Project Ben-Yehuda public-domain dump, Dicta's vowelling model |
| `wikipedia_articles.tsv` | The 271 Hebrew Wikipedia articles (with revision) whose first 4,000 paragraphs Dicta vowelled for the "modern vowelled" lines |
| `benyehuda_works.tsv` | The Project Ben-Yehuda works in the vowelled pool (those with dense niqqud) |
| `fonts.tsv` | The 165 font files used to render synthetic lines, with sha256, licence and source |
| `label_changes.tsv` | The 349 real lines whose training label differs from the Wikisource text, with both texts and the reason |

## What each model was trained on

Both models continued training from their 1.0 versions with Tesseract's `lstmtraining` (fine-tuning
only, learning rate 1e-4). Two kinds of lines:

**Real printed lines.** Lines cut from the 14 books in `books.tsv` (public-domain books printed
between 1876 and 1926, plus one 2025 book its author released into the public domain), labelled with the proofread Hebrew
Wikisource transcription of each page. `heb_rivok` saw 15,663 distinct lines, `heb_rivok_lite`
14,291. Three volumes of Hyman's *Toldoth Tannaim veAmoraim* (1910) are most of them, and the
vowelled 1885 Seder Tefilot is the next largest. Lines whose transcription differs from the print
(modernised spelling, expanded abbreviations, editor's notes) were dropped, and so were all lines on
pages that Wikisource marks as validated (quality 4), because another benchmark draws its test pages
from those.

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
| Rare marks | Lines from the vowelled sources that carry the rarest marks (hataf qamats, qubuts, meteg, ...) |

`heb_rivok`: 60,000 iterations on 82,932 lines (25% real), then 20,000 iterations on a 20,000-line
mix (35% real). `heb_rivok_lite`: 60,000 iterations on 60,000 lines (35% real), without the lines
that contain Latin letters (the model has none). The released files average the weights of the last
checkpoints.

## Used only to choose checkpoints, not trained on

- 1,500 lines of *Milhamot Hashem* (1923) and 307 lines from 23 held-out pages of the Seder Tefilot,
  both from Hebrew Wikisource;
- 400 synthetic lines from the four held-out Tanakh books;
- 253 lines from private receipts (not published);
- 500 synthetic lines of 1.0's mix, as a check against forgetting.

None of the test sets in the model card was used for training or for choosing checkpoints.

## Not in this folder

The line images, their labels, and the code that cut and aligned them are not published. The scans
and transcriptions they come from are all fetched by the script. Write to ronen@rivoklabs.com if you
need the lines themselves.

## Licences

- **Scans:** public domain, per each file's licence tag on Wikimedia Commons (`books.tsv`).
- **Transcriptions:** Hebrew Wikisource, CC BY-SA 4.0. The Wikisource text quoted in `label_changes.tsv`
  is under the same licence.
- **Text sources:** Tanakh (MAM), CC BY-SA 4.0; Mishnah (Torat Emet) and Project Ben-Yehuda, public
  domain; Hebrew Wikipedia, CC BY-SA 4.0; Dicta's model, CC BY 4.0.
- **Fonts:** OFL-1.1, GPL with font exception, and the DejaVu licence (see `fonts.tsv`).

Credits are in `../../NOTICE`.
