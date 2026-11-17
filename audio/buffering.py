from __future__ import annotations

import time
from collections import deque

import numpy as np


class EphemeralAudioBuffer:
    """Bounded RAM-only audio buffer. It never serializes audio to a store."""

    def __init__(self, max_seconds: float, sample_rate: int, ttl_seconds: float = 60.0):
        if max_seconds <= 0 or sample_rate <= 0 or ttl_seconds <= 0:
            raise ValueError("buffer dimensions and TTL must be positive")
        self.capacity = int(max_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.ttl_seconds = ttl_seconds
        self._samples = deque(maxlen=self.capacity)
        self.last_write = time.monotonic()

    def append(self, samples: np.ndarray) -> None:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._samples.extend(values.tolist())
        self.last_write = time.monotonic()

    def read(self, count: int | None = None) -> np.ndarray:
        values = np.fromiter(self._samples, dtype=np.float32)
        return values if count is None else values[-count:].copy()

    def expire_if_stale(self, now: float | None = None) -> bool:
        if (now or time.monotonic()) - self.last_write >= self.ttl_seconds:
            self.clear()
            return True
        return False

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)


class SlidingWindowGenerator:
    def __init__(self, window_samples: int, hop_samples: int):
        if window_samples <= 0 or hop_samples <= 0:
            raise ValueError("window and hop must be positive")
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self._buffer = np.empty(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        self._buffer = np.concatenate((self._buffer, np.asarray(samples, dtype=np.float32).reshape(-1)))
        windows: list[np.ndarray] = []
        while self._buffer.size >= self.window_samples:
            windows.append(self._buffer[: self.window_samples].copy())
            self._buffer = self._buffer[self.hop_samples :]
        return windows

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
