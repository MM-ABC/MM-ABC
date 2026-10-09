"""Resolve configs and pretrained weights against the repository root."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("MMABC_HOME", REPO_ROOT)).resolve()


def home_path(path: str | os.PathLike) -> Path:
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else HOME / p


def repo_path(path: str | os.PathLike) -> Path:
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else REPO_ROOT / p
