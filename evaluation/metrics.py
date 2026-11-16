from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(labels: list[int], scores: list[float], threshold: float = 0.5) -> dict[str, float | list]:
    y, p = np.asarray(labels), np.asarray(scores)
    predicted = (p >= threshold).astype(int)
    result: dict[str, float | list] = {"precision": float(precision_score(y, predicted, zero_division=0)), "recall": float(recall_score(y, predicted, zero_division=0)), "f1": float(f1_score(y, predicted, zero_division=0)), "accuracy": float(accuracy_score(y, predicted)), "brier_score": float(brier_score_loss(y, p)), "confusion_matrix": confusion_matrix(y, predicted).tolist()}
    if len(np.unique(y)) == 2: result["roc_auc"] = float(roc_auc_score(y, p))
    return result
