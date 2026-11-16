from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(42); signal=np.asarray(audio,dtype=np.float32); power=np.mean(signal*signal)+1e-12; noise_power=power/(10**(snr_db/10)); return (signal+rng.normal(0,np.sqrt(noise_power),signal.shape)).astype(np.float32)


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    return resample_poly(np.asarray(audio,dtype=np.float32), target_rate, source_rate).astype(np.float32)


def bandwidth_limit(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 3400) -> np.ndarray:
    spectrum=np.fft.rfft(np.asarray(audio,dtype=np.float32)); frequencies=np.fft.rfftfreq(len(audio),1/sample_rate); spectrum[frequencies>cutoff_hz]=0; return np.fft.irfft(spectrum,n=len(audio)).astype(np.float32)


def packet_loss(audio: np.ndarray, packet_samples: int, loss_rate: float, seed: int = 42) -> np.ndarray:
    if not 0 <= loss_rate <= 1 or packet_samples <= 0: raise ValueError("invalid packet-loss parameters")
    result=np.asarray(audio,dtype=np.float32).copy(); rng=np.random.default_rng(seed)
    for start in range(0,len(result),packet_samples):
        if rng.random() < loss_rate: result[start:start+packet_samples]=0
    return result
