from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.fft import dct

from audio.validation import validate_audio
from .fourier import stft


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray) -> np.ndarray: return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel: np.ndarray) -> np.ndarray: return 700 * (10 ** (mel / 2595) - 1)
    points = mel_to_hz(np.linspace(hz_to_mel(np.array([0]))[0], hz_to_mel(np.array([sample_rate / 2]))[0], n_mels + 2))
    bins = np.floor((n_fft + 1) * points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i : i + 3]
        if center > left: filters[i, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        if right > center: filters[i, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    return filters


def extract_features(audio: np.ndarray, sample_rate: int = 16000, selected: Iterable[str] | None = None, mode: str = "fast") -> dict[str, float]:
    values = validate_audio(audio, sample_rate)
    n_fft, hop = 512, max(1, sample_rate // 100)
    spec = np.abs(stft(values, n_fft=n_fft, hop_length=hop)) + 1e-10
    power = spec * spec
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    weights = power / np.sum(power, axis=0, keepdims=True)
    centroid = np.sum(freqs[:, None] * weights, axis=0)
    bandwidth = np.sqrt(np.sum(((freqs[:, None] - centroid) ** 2) * weights, axis=0))
    cumulative = np.cumsum(power, axis=0)
    rolloff_idx = np.argmax(cumulative >= 0.85 * cumulative[-1:], axis=0)
    flatness = np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)
    flux = np.sqrt(np.mean(np.diff(power, axis=1, prepend=power[:, :1]) ** 2, axis=0))
    entropy = -np.sum(weights * np.log(weights), axis=0) / np.log(weights.shape[0])
    zcr = np.mean(np.abs(np.diff(np.signbit(values))))
    mel = _mel_filterbank(sample_rate, n_fft, 20) @ power
    mfcc = dct(np.log(mel + 1e-10), type=2, axis=0, norm="ortho")[:13]
    chunks = np.array_split(values, max(1, values.size // hop))
    voiced = np.mean([float(np.sqrt(np.mean(chunk * chunk))) > 0.01 for chunk in chunks])
    result = {
        "spectral_centroid_mean": float(np.mean(centroid)), "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)), "spectral_rolloff_85_mean": float(np.mean(freqs[rolloff_idx])),
        "spectral_flatness_mean": float(np.mean(flatness)), "spectral_flux_mean": float(np.mean(flux)),
        "spectral_entropy_mean": float(np.mean(entropy)), "zero_crossing_rate": float(zcr),
        "high_frequency_ratio": float(np.mean(power[freqs >= min(4000, sample_rate * 0.25)]) / np.mean(power)),
        "voiced_ratio_proxy": float(voiced), "energy_mean": float(np.mean(values * values)),
        "energy_std": float(np.std(values * values)), "pitch_proxy_hz": float(_pitch_proxy(values, sample_rate)),
        "mfcc_mean_0": float(np.mean(mfcc[0])), "mfcc_mean_1": float(np.mean(mfcc[1])),
        "pause_ratio_proxy": float(1 - voiced),
    }
    if mode == "full":
        phase = np.unwrap(np.angle(stft(values, n_fft=n_fft, hop_length=hop)), axis=1)
        result.update({"phase_consistency": float(np.mean(np.cos(np.diff(phase, axis=1)))), "phase_discontinuity": float(np.mean(np.abs(np.diff(phase, axis=1))))})
    if selected is not None:
        result = {name: result[name] for name in selected if name in result}
    return {key: float(value) if np.isfinite(value) else 0.0 for key, value in result.items()}


def _pitch_proxy(audio: np.ndarray, sample_rate: int) -> float:
    centered = audio - np.mean(audio)
    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    lo, hi = max(1, sample_rate // 500), max(2, sample_rate // 60)
    if hi >= corr.size: return 0.0
    lag = lo + int(np.argmax(corr[lo:hi]))
    return float(sample_rate / lag) if corr[lag] > 0 else 0.0
