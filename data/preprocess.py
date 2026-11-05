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


def pad_or_trim(audio: np.ndarray, target_num_samples: int) -> np.ndarray:
    """Return a copy of ``audio`` with exactly ``target_num_samples`` samples."""
    if target_num_samples <= 0:
        raise ValueError("target_num_samples must be positive")
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size >= target_num_samples:
        return samples[:target_num_samples].copy()
    return np.pad(samples, (0, target_num_samples - samples.size)).astype(np.float32)


def normalize_audio(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale audio to a bounded peak amplitude while preserving silence."""
    if peak <= 0:
        raise ValueError("peak must be positive")
    samples = np.asarray(audio, dtype=np.float32)
    maximum = float(np.max(np.abs(samples), initial=0.0))
    if maximum == 0.0:
        return samples.copy()
    return (samples * (peak / maximum)).astype(np.float32)


def log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16_000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Convert a waveform into a float32 log-Mel spectrogram."""
    if sample_rate <= 0 or n_mels <= 0 or n_fft <= 0 or hop_length <= 0:
        raise ValueError("spectrogram parameters must be positive")
    spectrogram = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    return librosa.power_to_db(spectrogram, ref=np.max).astype(np.float32)
