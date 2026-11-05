"""Discover labeled audio files and create reproducible dataset manifests."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
REAL_LABEL_NAMES = {"real", "human", "bonafide", "genuine"}
FAKE_LABEL_NAMES = {"fake", "deepfake", "spoof", "synthetic", "cloned"}


def discover_audio_files(dataset_dir: str | Path) -> list[Path]:
    """Return supported audio files below ``dataset_dir`` in stable order."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )
