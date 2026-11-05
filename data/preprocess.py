"""Audio loading and feature extraction utilities for deepfake detection."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(
    audio_path: str | Path,
    sample_rate: int = 16_000,
) -> tuple[np.ndarray, int]:
    """Load an audio file as mono floating-point samples at ``sample_rate``."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    audio, actual_rate = librosa.load(path, sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio file contains no samples: {path}")
    return audio.astype(np.float32, copy=False), actual_rate
