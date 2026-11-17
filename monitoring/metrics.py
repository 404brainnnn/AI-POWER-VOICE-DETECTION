from __future__ import annotations

import statistics
import time
from collections import defaultdict


class LatencyRegistry:
    """Process-local latency registry; export to CloudWatch/Prometheus later."""
    def __init__(self) -> None: self.samples: dict[str, list[float]] = defaultdict(list)
    def observe(self, name: str, milliseconds: float) -> None: self.samples[name].append(float(milliseconds))
    def snapshot(self) -> dict[str, dict[str, float | int]]:
        result = {}
        for name, values in self.samples.items():
            ordered = sorted(values)
            result[name] = {"count": len(values), "p50": ordered[int(.50 * (len(values)-1))], "p95": ordered[int(.95 * (len(values)-1))], "p99": ordered[int(.99 * (len(values)-1))], "mean": statistics.fmean(values)}
        return result


class Timer:
    def __init__(self, registry: LatencyRegistry, name: str): self.registry, self.name = registry, name
    def __enter__(self): self.started = time.perf_counter(); return self
    def __exit__(self, *_): self.registry.observe(self.name, (time.perf_counter() - self.started) * 1000)
