from __future__ import annotations

def fuse_scores(dsp_score: float | None, deep_score: float | None, alpha: float = 0.5) -> float | None:
    available = [(dsp_score, alpha), (deep_score, 1 - alpha)]
    present = [(score, weight) for score, weight in available if score is not None]
    if not present: return None
    total = sum(weight for _, weight in present)
    return float(sum(score * weight for score, weight in present) / total)
