from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def read_pcm_wav(path: str | Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Read PCM16/PCM32 WAV without requiring librosa or soundfile."""
    with wave.open(str(path), "rb") as handle:
        channels, width, rate, frames = handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.readframes(handle.getnframes())
    if width == 2: values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4: values = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else: raise ValueError("only PCM16 and PCM32 WAV are supported by the standard-library reader")
    values = values.reshape(-1, channels).mean(axis=1)
    if target_sample_rate and target_sample_rate != rate:
        values = resample_poly(values, target_sample_rate, rate).astype(np.float32); rate = target_sample_rate
    return np.clip(values, -1, 1), rate
