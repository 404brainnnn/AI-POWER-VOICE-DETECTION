import numpy as np
from training.calibration import calibrated_scores, choose_thresholds, expected_calibration_error
from training.evaluation import detection_metrics, eer


def test_metrics_and_thresholds_are_finite():
    labels=np.array([0,0,1,1]); scores=np.array([.1,.2,.8,.9])
    metrics=detection_metrics(scores,labels); assert 0 <= metrics["eer"] <= 1; assert 0 <= eer(scores,labels) <= 1
    thresholds=choose_thresholds(scores,labels); assert thresholds["no_action"] >= .2; assert 0 <= expected_calibration_error(scores,labels) <= 1
    assert np.allclose(calibrated_scores(np.array([[0.,1.],[1.,0.]]),1.0),[.7310586,.2689414],atol=1e-5)
