from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class ImpairmentConfig:
    packet_loss_rate: float = 0.0
    jitter_ms: float = 0.0
    reorder_rate: float = 0.0
    noise_snr_db: float | None = None
    bandwidth_hz: float | None = None
    gain: float = 1.0
    seed: int = 42


def impair_audio(audio: np.ndarray, sample_rate: int, config: ImpairmentConfig) -> np.ndarray:
    values=np.asarray(audio,dtype=np.float32).copy()*config.gain; rng=np.random.default_rng(config.seed)
    if config.noise_snr_db is not None:
        power=np.mean(values*values)+1e-12; values+=rng.normal(0,np.sqrt(power/(10**(config.noise_snr_db/10))),len(values)).astype(np.float32)
    if config.bandwidth_hz is not None and config.bandwidth_hz < sample_rate/2:
        spectrum=np.fft.rfft(values); frequencies=np.fft.rfftfreq(len(values),1/sample_rate); spectrum[frequencies>config.bandwidth_hz]=0; values=np.fft.irfft(spectrum,n=len(values)).astype(np.float32)
    return np.clip(values,-1,1)


def reorder_packets(packets: list, rate: float, seed: int) -> list:
    rng=random.Random(seed); result=[]; index=0
    while index<len(packets):
        if index+1<len(packets) and rng.random()<rate: result.extend([packets[index+1],packets[index]]); index+=2
        else: result.append(packets[index]); index+=1
    return result
