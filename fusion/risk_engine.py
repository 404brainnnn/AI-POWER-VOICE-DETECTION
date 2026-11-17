from __future__ import annotations

from dataclasses import dataclass

from .bayesian import SequentialBayesianRisk
from .weighted import fuse_scores


@dataclass(frozen=True)
class RiskDecision:
    fused_score: float | None
    posterior: float
    tier: str
    mode: str


class RiskEngine:
    def __init__(self, prior: float = 0.01, discount: float = 0.35, low: float = 0.20, high: float = 0.70):
        self.low, self.high = low, high
        self.bayesian = SequentialBayesianRisk(prior, discount)

    def update(self, dsp_score: float | None = None, deep_score: float | None = None, alpha: float = 0.5) -> RiskDecision:
        score = fuse_scores(dsp_score, deep_score, alpha)
        if score is None: return RiskDecision(None, self.bayesian.posterior, "insufficient_evidence", "no-score")
        posterior = self.bayesian.update(score)
        mode = "hybrid" if dsp_score is not None and deep_score is not None else "dsp-only" if dsp_score is not None else "deep-only"
        return RiskDecision(score, posterior, self.bayesian.classify(self.low, self.high), mode)

    def reset(self) -> None: self.bayesian.reset()
