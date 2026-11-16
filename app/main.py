from __future__ import annotations

import json
import time

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .schemas import AnalyzeRequest, AlertResponse, DetectionResponse, SessionResponse, now
from .services import DetectionService
from .core.security import require_api_key
from monitoring.metrics import LatencyRegistry

app = FastAPI(title="Voice Clone Impersonation Detector", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
service = DetectionService(settings.default_sample_rate)
latencies = LatencyRegistry()


@app.get("/health")
def health() -> dict[str, str]: return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]: return {"status": "ready", "deep_model": "not_configured"}


@app.get("/api/v1/system/status")
def status() -> dict: return {"service": "voice-clone-detector", "detection_mode": "dsp-only", "sessions": len(service.sessions), "raw_audio_persistence": False}


@app.get("/api/v1/models")
def models() -> dict: return {"models": [{"name": "dsp-baseline", "status": "available_for_training", "trained": False}, {"name": "deep-learning", "status": "deferred"}]}


@app.post("/api/v1/sessions", response_model=SessionResponse)
def create_session(sample_rate: int = settings.default_sample_rate, _: None = Depends(require_api_key)) -> SessionResponse:
    try: session = service.create_session(sample_rate)
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc
    return SessionResponse(session_id=session.session_id, sample_rate=sample_rate, created_at=now())


@app.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    if session_id not in service.sessions: raise HTTPException(404, "session not found")
    session = service.sessions[session_id]
    return {"session_id": session_id, "sample_rate": session.sample_rate, "buffered_samples": len(session.buffer), "risk": session.risk.bayesian.posterior}


@app.delete("/api/v1/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, _: None = Depends(require_api_key)) -> None:
    if session_id not in service.sessions: raise HTTPException(404, "session not found")
    service.delete_session(session_id)


@app.post("/api/v1/detection/analyze", response_model=list[DetectionResponse])
def analyze(request: AnalyzeRequest, _: None = Depends(require_api_key)) -> list[DetectionResponse]:
    session = service.sessions.get(request.session_id) if request.session_id else None
    if session is None: session = service.create_session(request.sample_rate)
    if session.sample_rate != request.sample_rate: raise HTTPException(400, "sample rate differs from session")
    started = time.perf_counter()
    try: results = service.analyze(session, np.asarray(request.audio, dtype=np.float32))
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    return [DetectionResponse(session_id=session.session_id, detection_mode=item["decision"].mode, dsp_score=item["dsp_score"], fused_score=item["decision"].fused_score, sequential_risk=item["decision"].posterior, risk_tier=item["decision"].tier, calibration_status="unavailable-untrained" if item["dsp_score"] is None else "unverified", processing_latency_ms=(time.perf_counter() - started) * 1000, observation_delay_ms=3000.0, model_versions=[item["model_version"]] if item["model_version"] else [], timestamp=now(), warning="No trained DSP model is configured; this result contains no synthetic score") for item in results]


@app.post("/api/v1/detection/batch", response_model=list[DetectionResponse])
def analyze_batch(request: AnalyzeRequest, _: None = Depends(require_api_key)) -> list[DetectionResponse]:
    return analyze(request)


@app.get("/api/v1/metrics")
def metrics() -> dict: return {"latency": latencies.snapshot(), "sessions": len(service.sessions), "raw_audio_persisted": False}


@app.get("/api/v1/alerts", response_model=list[AlertResponse])
def alerts() -> list[AlertResponse]: return [AlertResponse(**{**a, "created_at": now()}) for a in service.alerts[-100:]]


@app.get("/api/v1/alerts/{alert_id}", response_model=AlertResponse)
def get_alert(alert_id: str) -> AlertResponse:
    for alert in service.alerts:
        if alert["alert_id"] == alert_id: return AlertResponse(**{**alert, "created_at": now()})
    raise HTTPException(404, "alert not found")


@app.post("/api/v1/alerts/{alert_id}/acknowledge", response_model=AlertResponse)
def acknowledge_alert(alert_id: str, _: None = Depends(require_api_key)) -> AlertResponse:
    for alert in service.alerts:
        if alert["alert_id"] == alert_id:
            alert["acknowledged"] = True; return AlertResponse(**{**alert, "created_at": now()})
    raise HTTPException(404, "alert not found")


@app.websocket("/api/v1/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = service.sessions.get(session_id)
    if session is None:
        await websocket.close(code=1008, reason="unknown session"); return
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect": break
            payload = json.loads(message.get("text", "{}")) if message.get("text") else message.get("bytes", b"")
            if isinstance(payload, dict):
                audio = payload.get("audio", [])
                sample_rate = payload.get("sample_rate", session.sample_rate)
            else: audio, sample_rate = np.frombuffer(payload, dtype=np.float32).tolist(), session.sample_rate
            if sample_rate != session.sample_rate: await websocket.send_json({"error": "sample rate mismatch"}); continue
            results = service.analyze(session, np.asarray(audio, dtype=np.float32))
            for item in results: await websocket.send_json({"session_id": session_id, "dsp_score": item["dsp_score"], "risk": item["decision"].posterior, "tier": item["decision"].tier, "mode": item["decision"].mode, "processing_latency_ms": item["latency_ms"]})
    except WebSocketDisconnect: pass
