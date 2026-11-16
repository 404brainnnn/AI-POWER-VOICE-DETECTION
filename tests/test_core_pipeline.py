import numpy as np

from app.main import health, service
from audio.buffering import EphemeralAudioBuffer, SlidingWindowGenerator
from audio.vad import voice_activity_ratio
from dsp.features import extract_features
from fusion.bayesian import SequentialBayesianRisk
from fusion.weighted import fuse_scores


def test_fourier_features_are_finite_and_named():
    features = extract_features(np.sin(2 * np.pi * 220 * np.arange(48000) / 16000), 16000)
    assert "spectral_centroid_mean" in features
    assert all(np.isfinite(list(features.values())))


def test_buffer_and_sliding_windows_are_bounded():
    buffer = EphemeralAudioBuffer(1, 100)
    buffer.append(np.ones(150)); assert len(buffer) == 100
    gen = SlidingWindowGenerator(100, 50)
    assert len(gen.push(np.ones(150))) == 2
    assert len(gen.push(np.ones(50))) == 1


def test_vad_and_fusion_and_bayesian():
    assert voice_activity_ratio(np.ones(1600), -45, 16000) > 0.9
    assert fuse_scores(0.2, None) == 0.2
    risk = SequentialBayesianRisk(prior=0.1, discount=0.5)
    assert risk.update(0.9) > 0.1


def test_api_health_and_session():
    service.sessions.clear()
    assert health()["status"] == "ok"
    from app.main import create_session, get_session
    response = create_session(16000)
    assert get_session(response.session_id)["sample_rate"] == 16000
