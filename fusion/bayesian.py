from __future__ import annotations

import math


class SequentialBayesianRisk:
    def __init__(self, prior: float = 0.01, discount: float = 0.35, min_evidence: int = 1):
        if not 0 < prior < 1 or not 0 < discount <= 1: raise ValueError("prior and discount are invalid")
        self.prior, self.discount, self.min_evidence = prior, discount, min_evidence
        self.reset()

    def reset(self) -> None:
        self.log_odds = math.log(self.prior / (1 - self.prior))
        self.count = 0

    def update(self, calibrated_probability: float) -> float:
        if not 0 <= calibrated_probability <= 1: raise ValueError("score must be between 0 and 1")
        eps = 1e-6
        p = min(max(calibrated_probability, eps), 1 - eps)
        evidence = math.log(p / (1 - p))
        self.log_odds += self.discount * evidence
        self.count += 1
        return self.posterior

    @property
    def posterior(self) -> float:
        return float(1 / (1 + math.exp(-max(-60, min(60, self.log_odds)))))

    @property
    def sufficient_evidence(self) -> bool: return self.count >= self.min_evidence

    def classify(self, low: float = 0.20, high: float = 0.70) -> str:
        if not self.sufficient_evidence: return "insufficient_evidence"
        return "high" if self.posterior >= high else "medium" if self.posterior >= low else "low"
