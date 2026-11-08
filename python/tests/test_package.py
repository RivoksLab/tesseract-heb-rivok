"""Tests for tesseract-heb-rivok. The OCR tests need Tesseract on PATH and are skipped without it."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import tesseract_heb_rivok as thr

HAVE_TESS = shutil.which("tesseract") is not None


def test_files_match_published_hashes():
    assert thr.verify() == {name: True for name in thr.MODELS}


def test_tessdata_dir_holds_both_models():
    d = Path(thr.tessdata_dir())
    assert sorted(p.name for p in d.glob("*.traineddata")) == ["heb_rivok.traineddata", "heb_rivok_lite.traineddata"]


def test_merged_dir_links_system_and_models(tmp_path):
    sysdir = tmp_path / "sys"
    (sysdir / "configs").mkdir(parents=True)
    (sysdir / "configs" / "hocr").write_text("tessedit_create_hocr 1\n")
    (sysdir / "eng.traineddata").write_bytes(b"eng")
    (sysdir / "heb_rivok.traineddata").write_bytes(b"old copy")  # must be replaced by ours
    out = Path(thr.merged_tessdata_dir(tmp_path / "merged", system_dir=str(sysdir)))
    assert (out / "eng.traineddata").read_bytes() == b"eng"
    assert (out / "configs" / "hocr").is_file()
    assert (out / "heb_rivok.traineddata").resolve() == (Path(thr.tessdata_dir()) / "heb_rivok.traineddata").resolve()
    # calling again refreshes in place
    assert thr.merged_tessdata_dir(out, system_dir=str(sysdir)) == str(out)


def test_merged_dir_refuses_the_system_folder(tmp_path):
    with pytest.raises(ValueError):
        thr.merged_tessdata_dir(tmp_path, system_dir=str(tmp_path))


@pytest.mark.skipif(not HAVE_TESS, reason="tesseract not on PATH")
def test_tesseract_lists_both_models():
    out = subprocess.run(["tesseract", "--tessdata-dir", thr.tessdata_dir(), "--list-langs"],
                         capture_output=True, text=True, check=True).stdout.split()
    assert "heb_rivok" in out and "heb_rivok_lite" in out


@pytest.mark.skipif(not HAVE_TESS, reason="tesseract not on PATH")
def test_merged_dir_has_system_languages(tmp_path):
    if thr.system_tessdata_dir() is None:
        pytest.skip("no system tessdata")
    env = dict(os.environ, TESSDATA_PREFIX=thr.merged_tessdata_dir(tmp_path / "m"))
    langs = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, env=env, check=True).stdout.split()
    assert {"heb_rivok", "heb_rivok_lite", "osd"} <= set(langs)


@pytest.mark.skipif(not HAVE_TESS, reason="tesseract not on PATH")
def test_merged_dir_rebuilds_while_prefix_points_at_it(tmp_path, monkeypatch):
    if thr.system_tessdata_dir(use_env=False) is None:
        pytest.skip("no system tessdata")
    out = thr.merged_tessdata_dir(tmp_path / "m")
    monkeypatch.setenv("TESSDATA_PREFIX", out)
    assert thr.merged_tessdata_dir(tmp_path / "m") == out


def test_cli_prints_folder(capsys):
    from tesseract_heb_rivok.__main__ import main
    assert main([]) == 0
    assert capsys.readouterr().out.strip() == thr.tessdata_dir()
