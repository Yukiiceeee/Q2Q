"""Project-root-relative resolution for model and dataset locations.

Nothing in this repository may hardcode an absolute path: a fresh clone has to
run without anyone editing a config file first. Every model and dataset
location is therefore written relative to the project root, and this module is
the single place that turns such a relative path into an absolute one.

Set MODELS_DIR / DATA_DIR when the real assets live outside the checkout, which
is the normal case for a shared model cache or a mounted dataset volume. Both
are read on every call rather than captured at import time, so a ``.env`` that
is loaded later in startup still takes effect.
"""
from __future__ import annotations

import os
from pathlib import Path

# src/paths.py -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def models_dir() -> Path:
    """Root holding local model weights. Override with the MODELS_DIR env var."""
    return Path(os.environ.get("MODELS_DIR") or PROJECT_ROOT / "models")


def data_dir() -> Path:
    """Root holding datasets. Override with the DATA_DIR env var."""
    return Path(os.environ.get("DATA_DIR") or PROJECT_ROOT / "data")


def model_path(name: str) -> str:
    """Absolute path of one model directory, e.g. ``model_path("bge-m3")``."""
    return str(models_dir() / name)


def dataset_path(relative: str | Path) -> Path:
    """Absolute path of one dataset file, given its path under ``data_dir()``."""
    return data_dir() / relative


def resolve(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a config-supplied path against ``root`` (default: project root).

    An already-absolute path is returned untouched, so an operator can still
    point a config at an out-of-tree location without this module getting in
    the way.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root or PROJECT_ROOT) / candidate
