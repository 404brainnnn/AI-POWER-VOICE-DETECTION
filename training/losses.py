from __future__ import annotations


def weighted_cross_entropy(class_counts: list[int]):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc: raise RuntimeError("install PyTorch from requirements-ml.txt") from exc
    if len(class_counts) != 2 or min(class_counts) <= 0: raise ValueError("both classes must occur in training data")
    weights=torch.tensor([sum(class_counts)/(2*count) for count in class_counts],dtype=torch.float32)
    return weights, lambda logits, labels: F.cross_entropy(logits,labels,weight=weights.to(logits.device))


class OCSoftmaxLoss:
    """Compact one-class softmax option; configurable, but not enabled by default."""
    def __init__(self, margin: float = 0.35, scale: float = 20.0): self.margin,self.scale=margin,scale
    def __call__(self, logits, labels):
        import torch
        real_logit=logits[:,0]; spoof_logit=logits[:,1]
        target=torch.where(labels==1, spoof_logit-self.margin, real_logit-self.margin)
        other=torch.where(labels==1, real_logit, spoof_logit)
        return torch.nn.functional.softplus(self.scale*(other-target)).mean()
