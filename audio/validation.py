from __future__ import annotations

import numpy as np


def validate_audio(samples: np.ndarray, sample_rate: int, max_samples: int = 480_000) -> np.ndarray:
    if sample_rate not in {8000, 16000, 48000}:
        raise ValueError("sample_rate must be one of 8000, 16000, or 48000")
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0 or values.size > max_samples:
        raise ValueError("audio must be non-empty and within the configured size limit")
    if not np.isfinite(values).all():
        raise ValueError("audio contains non-finite values")
    return np.clip(values, -1.0, 1.0)
