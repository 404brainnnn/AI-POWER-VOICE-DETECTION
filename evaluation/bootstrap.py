from __future__ import annotations

import numpy as np


def grouped_bootstrap(values: list[float], groups: list[str], statistic=np.mean, iterations: int = 1000, seed: int = 42) -> tuple[float, float]:
    if len(values) != len(groups) or not values: raise ValueError("values and groups must be non-empty and aligned")
    unique=list(dict.fromkeys(groups)); rng=np.random.default_rng(seed); estimates=[]
    for _ in range(iterations):
        selected=rng.choice(unique,size=len(unique),replace=True); sampled=[value for group in selected for value, actual in zip(values,groups) if actual==group]; estimates.append(float(statistic(sampled)))
    return float(np.percentile(estimates,2.5)), float(np.percentile(estimates,97.5))
