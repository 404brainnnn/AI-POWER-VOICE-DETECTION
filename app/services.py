from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from audio.buffering import EphemeralAudioBuffer, SlidingWindowGenerator
from audio.validation import validate_audio
from audio.vad import is_speech
from dsp.features import extract_features
from fusion.risk_engine import RiskEngine
from models.dsp_detector import DSPDetector


@dataclass
class Session:
    session_id: str
    sample_rate: int
    created_at: float = field(default_factory=time.time)
    buffer: EphemeralAudioBuffer = field(init=False)
    windows: SlidingWindowGenerator = field(init=False)
    risk: RiskEngine = field(default_factory=RiskEngine)

    def __post_init__(self) -> None:
        self.buffer = EphemeralAudioBuffer(30, self.sample_rate)
        self.windows = SlidingWindowGenerator(self.sample_rate * 3, self.sample_rate)


class DetectionService:
    def __init__(self, sample_rate: int = 16000, detector: DSPDetector | None = None):
        self.sample_rate = sample_rate
        self.detector = detector
        self.sessions: dict[str, Session] = {}
        self.alerts: list[dict] = []

    def create_session(self, sample_rate: int) -> Session:
        if sample_rate not in {8000, 16000, 48000}: raise ValueError("unsupported sample rate")
        session = Session(str(uuid.uuid4()), sample_rate)
        self.sessions[session.session_id] = session
        return session

    def delete_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session: session.buffer.clear(); session.windows.reset(); session.risk.reset()

    def analyze(self, session: Session, audio: np.ndarray) -> list[dict]:
        started = time.perf_counter()
        values = validate_audio(audio, session.sample_rate)
        session.buffer.append(values)
        results = []
        for window in session.windows.push(values):
            if not is_speech(window): continue
            features = extract_features(window, session.sample_rate)
            dsp_output = self.detector.predict_proba(window, session.sample_rate) if self.detector else None
            decision = session.risk.update(dsp_score=dsp_output.score if dsp_output else None)
            result = {"dsp_score": dsp_output.score if dsp_output else None, "decision": decision, "features": features,
                      "model_version": dsp_output.model_version if dsp_output else None,
                      "latency_ms": (time.perf_counter() - started) * 1000}
            results.append(result)
            if decision.tier == "high": self.alerts.append({"alert_id": str(uuid.uuid4()), "session_id": session.session_id, "tier": "high", "message": "High voice-cloning risk; step-up verification recommended", "created_at": time.time(), "acknowledged": False})
        return results
