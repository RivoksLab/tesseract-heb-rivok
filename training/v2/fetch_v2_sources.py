#!/usr/bin/env python3
"""Download every document and source file listed in this folder: the inputs of heb_rivok 2.0
and heb_rivok_lite 2.0. Standard library only. Nothing is redistributed by this repository;
everything comes from the original sites, and each file is checked against the hash in the
manifest.

  books       the 14 scanned books, from Wikimedia Commons (books.tsv; checked by sha1)       ~406 MB
  pages       the 522 Hebrew Wikisource pages whose lines were trained on, at the exact
              revision used (pages.tsv), as wikitext
  text        Tanakh (MAM), Mishnah (Torat Emet), Project Ben-Yehuda dump (text_sources.tsv)  ~281 MB
  wikipedia   the 271 Hebrew Wikipedia articles whose paragraphs Dicta vowelled, at the
              revision used (wikipedia_articles.tsv), as wikitext
  fonts       the font files (fonts.tsv), checked by sha256; system fonts are only checked
  dicta       Dicta's vowelling model at the pinned revision (only with --with-dicta)          ~1.2 GB

    python3 fetch_v2_sources.py [--out v2_sources] [--only books,pages,...] [--with-dicta]

Wiki API calls are batched (50 per request), sent one at a time, one second apart, with maxlag.
Some differences from our run are expected and reported, not fatal: a file that changed at its
source (for example a branch URL). Three Noto font files whose download was not recorded are not
fetched: the line dataset carries them (fonts/, OFL-1.1).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = "heb_rivok-v2-sources/1.0 (+https://github.com/RivoksLab/tesseract-heb-rivok)"
COMMONS = "https://commons.wikimedia.org/w/api.php"
WIKISOURCE = "https://he.wikisource.org/w/api.php"
WIKIPEDIA = "https://he.wikipedia.org/w/api.php"
csv.field_size_limit(1 << 30)
problems = []


def rows(name):
    with open(HERE / name, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def http(url, data=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:                                  # noqa: BLE001 (network: retry, then fail)
            if attempt == 4:
                raise
            print(f"  retry {attempt + 1} ({e})", file=sys.stderr)
            time.sleep(5 * (attempt + 1))


_last = [0.0]


def api(endpoint, **params):
    wait = 1.0 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    params.update(format="json", formatversion="2", maxlag="5")
    out = json.loads(http(endpoint, urllib.parse.urlencode(params).encode()))
    _last[0] = time.time()
    if "error" in out:
        raise RuntimeError(f"{endpoint}: {out['error']}")
    return out


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dest, algo=None, want=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and want and digest(dest, algo) == want):
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(http(url))
        tmp.replace(dest)
    if want and digest(dest, algo) != want:
        problems.append(f"{dest}: {algo} differs from the manifest (the file changed at its source)")


def batches(items, n=50):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def books(out):
    bs = rows("books.tsv")
    for group in batches(bs):
        info = api(COMMONS, action="query", prop="imageinfo", iiprop="url|sha1|size", iilimit="50",
                   titles="|".join(b["commons_file"] for b in group))
        norm = {n["from"]: n["to"] for n in info["query"].get("normalized", [])}
        pages = {p["title"]: p for p in info["query"]["pages"]}
        for b in group:
            p = pages.get(norm.get(b["commons_file"], b["commons_file"]), {})
            versions = p.get("imageinfo", [])
            match = next((v for v in versions if v["sha1"] == b["sha1"]), None)
            if not match:
                problems.append(f"{b['commons_file']}: no version with sha1 {b['sha1']} on Commons")
                continue
            dest = out / "books" / (b["book"] + Path(b["commons_file"]).suffix.lower())
            print(f"book {b['book']} ({int(b['bytes']) / 1e6:.0f} MB)")
            download(match["url"], dest, "sha1", b["sha1"])


def wiki_revisions(endpoint, revids, dest_of):
    for group in batches(revids):
        r = api(endpoint, action="query", prop="revisions", revids="|".join(group),
                rvprop="ids|content", rvslots="main")
        got = set()
        for p in r["query"].get("pages", []):
            for rev in p.get("revisions", []):
                d = dest_of[str(rev["revid"])]
                d.parent.mkdir(parents=True, exist_ok=True)
                d.write_text(rev["slots"]["main"]["content"], encoding="utf-8")
                got.add(str(rev["revid"]))
        for rv in set(group) - got:
            problems.append(f"{endpoint}: revision {rv} not returned (deleted or hidden?)")
        print(f"  {len(got)}/{len(group)} revisions")


def pages(out):
    ps = rows("pages.tsv")
    print(f"Wikisource: {len(ps)} pages")
    wiki_revisions(WIKISOURCE, [p["revid"] for p in ps],
                   {p["revid"]: out / "pages" / p["book"] / f"{int(p['pdf_page']):04d}.wiki" for p in ps})


def wikipedia(out):
    arts = rows("wikipedia_articles.tsv")
    print(f"Wikipedia: {len(arts)} articles")
    wiki_revisions(WIKIPEDIA, [a["revid"] for a in arts],
                   {a["revid"]: out / "wikipedia" / f"{a['pageid']}.wiki" for a in arts})


def text(out, with_dicta):
    for t in rows("text_sources.tsv"):
        if t["source"] == "wikipedia" or not t["url"].startswith("http") or " " in t["url"]:
            continue                                            # our own API dump: see wikipedia_articles.tsv
        if t["source"] == "dicta" and not with_dicta:
            continue
        if "not trained on" in t["used_for"]:
            continue
        print(f"{t['source']}: {t['path']}")
        download(t["url"], out / "text" / t["path"], "sha256", t["sha256"] or None)


def fonts(out):
    archives = {}
    for f in rows("fonts.tsv"):
        src = f["source"]
        if src.startswith("/usr/share/fonts"):
            path = src.split(" (")[0]
            pkg = src.split("package ")[-1].rstrip(")")
            if not os.path.exists(path):
                problems.append(f"{f['file']}: not installed; install the Debian/Ubuntu package {pkg}")
            elif digest(path, "sha256") != f["sha256"]:
                problems.append(f"{f['file']}: installed version differs from ours ({pkg})")
            continue
        if src.startswith("dataset:"):                          # our copy is in the line dataset (fonts/)
            continue
        if ": http" in src:                                     # a file inside an archive (Culmus, Ezra SIL)
            name, url = src.split(": ", 1)
            if url not in archives:
                print(f"font archive {name}")
                archives[url] = out / "fonts" / "archives" / name
                download(url, archives[url])
            continue
        print(f"font {f['file']}")
        download(src, out / "fonts" / f["file"], "sha256", f["sha256"])
    if archives:
        print("Culmus and Ezra SIL: unpack the archives in fonts/archives; fonts.tsv has each file's sha256.")
    print("Three Noto files whose download was not recorded are in the line dataset (fonts/);"
          " synth/collect_fonts.py gathers everything into one folder.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("v2_sources"))
    ap.add_argument("--only", default="books,pages,text,wikipedia,fonts")
    ap.add_argument("--with-dicta", action="store_true", help="also download Dicta's model (~1.2 GB)")
    a = ap.parse_args()
    parts = a.only.split(",")
    for part in parts:
        if part == "text":
            text(a.out, a.with_dicta)
        elif part in ("books", "pages", "wikipedia", "fonts"):
            globals()[part](a.out)
        else:
            sys.exit(f"unknown part {part}")
    print("\nDone." if not problems else "\nDone, with differences from our run:")
    for p in problems:
        print("  -", p)


if __name__ == "__main__":
    main()
