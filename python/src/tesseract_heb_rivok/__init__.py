"""heb_rivok and heb_rivok_lite, Tesseract models for printed Hebrew, installed with pip.

The models ship inside this package. Point Tesseract at them with ``tessdata_dir()``, or,
when you also need Tesseract's own models (``eng``, ``osd``) and config files, with
``merged_tessdata_dir()``.

    import tesseract_heb_rivok
    tesseract_heb_rivok.tessdata_dir()          # folder with heb_rivok + heb_rivok_lite
    tesseract_heb_rivok.merged_tessdata_dir()   # same, plus links to the system's tessdata
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

__version__ = "2.0.0"

#: Model name -> (model version, SHA-256 of the shipped integer ("fast") file).
MODELS = {
    "heb_rivok": ("2.0", "09ef1830f3393c590791abf9c18deca14450c8e5d3b556249d63b0183cf7a426"),
    "heb_rivok_lite": ("2.0", "59da073ad0bca9449fa5a965996e49d0e2bee3c17e8b0b4e2769d67a7ce34c40"),
}

_HERE = Path(__file__).resolve().parent


def tessdata_dir() -> str:
    """Folder holding ``heb_rivok.traineddata`` and ``heb_rivok_lite.traineddata``.

    Pass it to Tesseract as ``--tessdata-dir``. It holds only these two models, so
    ``-l heb_rivok+eng``, OSD and output configs such as ``hocr`` need
    :func:`merged_tessdata_dir` instead.
    """
    return str(_HERE / "tessdata")


def system_tessdata_dir(tesseract_cmd: str = "tesseract", use_env: bool = True) -> str | None:
    """The tessdata folder the installed Tesseract uses, or None if it can't be found.

    ``$TESSDATA_PREFIX`` wins when set (unless ``use_env=False``); otherwise the folder is
    read from ``tesseract --list-langs``.
    """
    env = os.environ.get("TESSDATA_PREFIX")
    if use_env and env and Path(env).is_dir():
        return str(Path(env))
    run_env = {k: v for k, v in os.environ.items() if k != "TESSDATA_PREFIX"}
    try:
        out = subprocess.run([tesseract_cmd, "--list-langs"], capture_output=True, text=True,
                             timeout=30, env=run_env)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'"([^"]+)"', out.stdout + out.stderr)
    return m.group(1).rstrip("/\\") if m and Path(m.group(1)).is_dir() else None


def _default_cache() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "tesseract-heb-rivok" / __version__ / "tessdata"


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:  # no symlink rights (e.g. Windows without developer mode): copy
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)


def merged_tessdata_dir(dest: str | os.PathLike | None = None, system_dir: str | None = None,
                        tesseract_cmd: str = "tesseract") -> str:
    """A tessdata folder with this package's models and everything in the system's tessdata.

    Every entry of the system folder (``eng``, ``osd``, ``configs``, ...) is linked in
    (copied where links aren't allowed), next to links to the two models here. Use it as
    ``TESSDATA_PREFIX`` for tools that need several languages, OSD or Tesseract's output
    configs: OCRmyPDF, Docling, ``-l heb_rivok+eng``.

    ``dest`` defaults to a per-user cache folder; calling again refreshes it.
    """
    out = Path(dest) if dest is not None else _default_cache()
    out.mkdir(parents=True, exist_ok=True)
    sysdir = system_dir or system_tessdata_dir(tesseract_cmd)
    if system_dir is None and sysdir is not None and Path(sysdir).resolve() == out.resolve():
        # TESSDATA_PREFIX already points at this merged folder: ask Tesseract without it
        sysdir = system_tessdata_dir(tesseract_cmd, use_env=False)
    if sysdir is None:
        raise RuntimeError("Tesseract's tessdata folder was not found: install Tesseract, "
                           "or pass system_dir=, or set TESSDATA_PREFIX")
    sysdir_p = Path(sysdir).resolve()
    if sysdir_p == out.resolve():
        raise ValueError("dest must not be the system tessdata folder")
    for entry in sysdir_p.iterdir():
        if entry.name.split(".")[0] not in MODELS:
            _link(entry, out / entry.name)
    for name in MODELS:
        _link(Path(tessdata_dir()) / f"{name}.traineddata", out / f"{name}.traineddata")
    return str(out)


def verify() -> dict[str, bool]:
    """Check the shipped model files against their published SHA-256."""
    ok = {}
    for name, (_, sha) in MODELS.items():
        p = Path(tessdata_dir()) / f"{name}.traineddata"
        ok[name] = p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == sha
    return ok


__all__ = ["MODELS", "tessdata_dir", "system_tessdata_dir", "merged_tessdata_dir", "verify", "__version__"]
