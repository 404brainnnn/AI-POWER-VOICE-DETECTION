from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    session_id: str
    sample_rate: int
    created_at: datetime


class AnalyzeRequest(BaseModel):
    session_id: str | None = None
    sample_rate: int = Field(16000, description="8000, 16000, or 48000 Hz")
    audio: list[float] = Field(min_length=1, max_length=480000)


class DetectionResponse(BaseModel):
    session_id: str
    detection_mode: str
    dsp_score: float | None = None
    deep_score: float | None = None
    fused_score: float | None = None
    sequential_risk: float
    risk_tier: str
    calibration_status: str
    processing_latency_ms: float
    observation_delay_ms: float
    model_versions: list[str]
    timestamp: datetime
    warning: str | None = None


class AlertResponse(BaseModel):
    alert_id: str
    session_id: str
    tier: str
    message: str
    created_at: datetime
    acknowledged: bool = False


class ScoreResponse(BaseModel):
    score: float | None = None
    tier: str
    model_version: str
    latency_ms: float


class CallCreateRequest(BaseModel):
    to: str = Field(min_length=8, max_length=16, pattern=r"^\+[1-9]\d{7,14}$")
    mode: Literal["human", "synthetic"]
    consent_confirmed: bool = False


class CallResponse(BaseModel):
    call_id: str
    status: str
    mode: str
    to: str
    provider: str = "dotvoice"


class CallEventResponse(BaseModel):
    event: str
    call_id: str
    status: str | None = None
    risk_tier: str | None = None
    score: float | None = None
    timestamp: datetime


def now() -> datetime:
    return datetime.now(timezone.utc)
