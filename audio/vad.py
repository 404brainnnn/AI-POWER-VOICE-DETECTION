from __future__ import annotations

import numpy as np


def frame_rms_db(audio: np.ndarray, frame_length: int = 320, hop_length: int = 160) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if values.size < frame_length:
        values = np.pad(values, (0, frame_length - values.size))
    frames = np.lib.stride_tricks.sliding_window_view(values, frame_length)[::hop_length]
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return (20 * np.log10(rms)).astype(np.float32)


def voice_activity_ratio(audio: np.ndarray, threshold_db: float = -45.0, sample_rate: int = 16000) -> float:
    db = frame_rms_db(audio, max(1, int(sample_rate * 0.02)), max(1, int(sample_rate * 0.01)))
    return float(np.mean(db > threshold_db))


def is_speech(audio: np.ndarray, threshold_db: float = -45.0, min_ratio: float = 0.1) -> bool:
    return voice_activity_ratio(audio, threshold_db) >= min_ratio
