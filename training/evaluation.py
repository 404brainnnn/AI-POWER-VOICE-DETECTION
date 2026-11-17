from __future__ import annotations

import numpy as np


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    order=np.argsort(scores)[::-1]; y=labels[order]; positives=max(1,int((labels==1).sum())); negatives=max(1,int((labels==0).sum())); fp=0; fn=positives; best=(1.0,1.0); best_gap=float("inf")
    for value in y:
        if value==1: fn-=1
        else: fp+=1
        fpr=fp/negatives; fnr=fn/positives
        gap=abs(fpr-fnr)
        if gap<best_gap: best=(fpr,fnr); best_gap=gap
    return float(sum(best)/2)


def min_dcf(scores: np.ndarray, labels: np.ndarray, prior: float = 0.05, c_miss: float = 1.0, c_fa: float = 1.0) -> float:
    thresholds=np.r_[0.0,np.sort(scores),1.0]; positives=max(1,(labels==1).sum()); negatives=max(1,(labels==0).sum()); values=[]
    for threshold in thresholds:
        pred=scores>=threshold; fn=((labels==1)&~pred).sum()/positives; fp=((labels==0)&pred).sum()/negatives; values.append(prior*c_miss*fn+(1-prior)*c_fa*fp)
    return float(min(values))


def detection_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str,float]:
    return {"eer":eer(scores,labels),"min_dcf":min_dcf(scores,labels),"false_alarm_at_90_recall":float(((scores[labels==0]>=np.quantile(scores[labels==1],.10)).mean()) if (labels==0).any() and (labels==1).any() else 0.0),"false_alarm_at_95_recall":float(((scores[labels==0]>=np.quantile(scores[labels==1],.05)).mean()) if (labels==0).any() and (labels==1).any() else 0.0),"false_alarm_at_99_recall":float(((scores[labels==0]>=np.quantile(scores[labels==1],.01)).mean()) if (labels==0).any() and (labels==1).any() else 0.0),"positive_rate":float(np.mean(labels==1)),"score_min":float(np.min(scores)),"score_max":float(np.max(scores))}
