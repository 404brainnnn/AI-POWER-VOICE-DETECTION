from __future__ import annotations

import numpy as np


def fft_magnitude(audio: np.ndarray, n_fft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = n_fft or values.size
    if n <= 0:
        raise ValueError("n_fft must be positive")
    spectrum = np.fft.rfft(values * np.hanning(values.size), n=n)
    return np.fft.rfftfreq(n, 1.0), np.abs(spectrum).astype(np.float32)


def stft(audio: np.ndarray, n_fft: int = 512, hop_length: int = 160, win_length: int | None = None) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    win = win_length or n_fft
    if min(n_fft, hop_length, win) <= 0:
        raise ValueError("STFT parameters must be positive")
    if values.size < win:
        values = np.pad(values, (0, win - values.size))
    window = np.hanning(win)
    frames = np.lib.stride_tricks.sliding_window_view(values, win)[::hop_length]
    if frames.size == 0:
        frames = values[None, :win]
    return np.fft.rfft(frames * window, n=n_fft, axis=1).T.astype(np.complex64)
