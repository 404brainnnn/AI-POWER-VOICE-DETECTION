from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def temperature_scale(logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
    try:
        import torch
        from torch.optim import LBFGS
    except ImportError as exc: raise RuntimeError("install PyTorch from requirements-ml.txt") from exc
    values=torch.tensor(logits,dtype=torch.float32); targets=torch.tensor(labels,dtype=torch.long); temperature=torch.nn.Parameter(torch.ones(())); optimizer=LBFGS([temperature],lr=0.01,max_iter=max_iter)
    def closure():
        optimizer.zero_grad(); loss=torch.nn.functional.cross_entropy(values/temperature.clamp_min(1e-3),targets); loss.backward(); return loss
    optimizer.step(closure); return float(temperature.detach().clamp_min(1e-3))


def calibrated_scores(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted=logits/temperature; shifted=shifted-shifted.max(axis=1,keepdims=True); probabilities=np.exp(shifted); probabilities/=probabilities.sum(axis=1,keepdims=True); return probabilities[:,1]


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    edges=np.linspace(0,1,bins+1); total=0.0
    for low,high in zip(edges[:-1],edges[1:]):
        mask=(scores>=low)&(scores<=high if high==1 else scores<high)
        if mask.any(): total += mask.mean()*abs(scores[mask].mean()-labels[mask].mean())
    return float(total)


def brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(scores, dtype=float)-np.asarray(labels, dtype=float))**2))


def choose_thresholds(scores: np.ndarray, labels: np.ndarray, target_false_alarm: float = 0.01) -> dict[str,float]:
    bona=scores[labels==0]
    if bona.size == 0: raise ValueError("held-out calibration set needs bona fide samples")
    warning=float(np.quantile(bona,1-target_false_alarm)); silent=float(np.quantile(bona,1-min(0.05,target_false_alarm*5)))
    return {"no_action": 0.0, "silent_flag": silent, "play_warning": max(silent,warning), "target_false_alarm": target_false_alarm}


def save_thresholds(path: str | Path, thresholds: dict[str,float]) -> None:
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(thresholds,indent=2),encoding="utf-8")
