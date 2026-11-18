from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .schemas import AnalyzeRequest, AlertResponse, CallCreateRequest, CallEventResponse, CallResponse, DetectionResponse, ScoreResponse, SessionResponse, now
from .services import DetectionService
from .core.security import require_api_key
from monitoring.metrics import LatencyRegistry
from audio.buffering import SlidingWindowGenerator
from audio.ingestion import decode_pcm16
from audio.vad import is_speech
from inference.queue import BatchingInferenceQueue
from monitoring.prometheus import metrics_text
from telephony.dotvoice import DotVoiceClient, DotVoiceConfig
import base64

app = FastAPI(title="Voice Clone Impersonation Detector", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
service = DetectionService(settings.default_sample_rate)
latencies = LatencyRegistry()
neural_runtime = None
inference_queue = None
startup_error = None
request_count = 0
tier_counts = {}
latency_samples = []
dotvoice = DotVoiceClient(DotVoiceConfig(settings.dotvoice_api_key,settings.dotvoice_from_number,settings.dotvoice_base_url,settings.public_base_url))
call_events: dict[str,list[dict]] = {}


@app.on_event("startup")
async def load_neural_runtime() -> None:
    global neural_runtime, inference_queue, startup_error
    if not settings.model_checkpoint: return
    try:
        from models.config import load_yaml_config
        from inference.runtime import NeuralRuntime
        if not Path(settings.model_checkpoint).is_file() or not Path(settings.thresholds_path).is_file(): raise FileNotFoundError("configured model checkpoint or thresholds file is missing")
        neural_runtime=NeuralRuntime.from_checkpoint(load_yaml_config(settings.model_config_path),settings.model_checkpoint,settings.thresholds_path,settings.model_device,settings.default_sample_rate)
        inference_queue=BatchingInferenceQueue(neural_runtime.predict_batch,settings.max_batch_size,settings.max_batch_wait_ms); await inference_queue.start()
    except Exception as exc: startup_error=str(exc)


@app.on_event("shutdown")
async def stop_inference_queue() -> None:
    if inference_queue: await inference_queue.stop()


@app.get("/health")
def health() -> dict[str, str]: return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]: return {"status": "ready", "deep_model": "not_configured"}


@app.get("/healthz")
def healthz() -> dict: return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    if settings.model_checkpoint and neural_runtime is None: raise HTTPException(503, startup_error or "model is not loaded")
    return {"status": "ready", "model_loaded": neural_runtime is not None}


@app.post("/score", response_model=ScoreResponse)
async def score(payload: bytes = Body(..., media_type="application/octet-stream"), x_sample_rate: int = Header(16000), _: None = Depends(require_api_key)) -> ScoreResponse:
    global request_count
    request_count += 1
    if inference_queue is None: raise HTTPException(503, "trained model runtime is not configured")
    try: waveform,sample_rate=decode_pcm16(payload,x_sample_rate,settings.default_sample_rate)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    if len(waveform) != settings.default_sample_rate*3: raise HTTPException(422,"/score requires exactly 3 seconds of PCM16 audio")
    result=await inference_queue.submit(waveform)
    tier_counts[result["tier"]]=tier_counts.get(result["tier"],0)+1; latency_samples.append(result["latency_ms"])
    return ScoreResponse(score=None if settings.hide_raw_scores else result["score"],tier=result["tier"],model_version=result["model_version"],latency_ms=result["latency_ms"])


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str: return metrics_text(request_count,tier_counts,latency_samples)


@app.get("/api/v1/system/status")
def status() -> dict: return {"service": "voice-clone-detector", "detection_mode": "dsp-only", "sessions": len(service.sessions), "raw_audio_persistence": False}


@app.get("/api/v1/models")
def models() -> dict: return {"models": [{"name": "dsp-baseline", "status": "available_for_training", "trained": False}, {"name": "deep-learning", "status": "deferred"}]}


@app.post("/api/v1/calls", response_model=CallResponse)
async def create_call(request: CallCreateRequest, _: None = Depends(require_api_key)) -> CallResponse:
    if not request.consent_confirmed: raise HTTPException(400,"confirm that the destination is authorized for this test call")
    try: provider_result=await dotvoice.create_call(request.to,request.mode,request.consent_confirmed)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    except RuntimeError as exc: raise HTTPException(503,str(exc)) from exc
    call_id=str(provider_result.get("call_id") or provider_result.get("id") or "unknown"); call_events.setdefault(call_id,[]).append({"event":"created","call_id":call_id,"status":provider_result.get("status","queued"),"timestamp":now()}); return CallResponse(call_id=call_id,status=provider_result.get("status","queued"),mode=request.mode,to=request.to)


@app.get("/api/v1/calls/{call_id}/events", response_model=list[CallEventResponse])
def get_call_events(call_id: str) -> list[CallEventResponse]: return [CallEventResponse(**event) for event in call_events.get(call_id,[])]


@app.post("/api/v1/telephony/dotvoice/answer")
def dotvoice_answer() -> dict: return {"status":"ok","stream_url":f"{settings.public_base_url.rstrip('/')}/api/v1/telephony/dotvoice/media"}


@app.post("/api/v1/telephony/dotvoice/webhook")
async def dotvoice_webhook(payload: dict) -> dict:
    call_id=str(payload.get("call_id") or payload.get("id") or "unknown"); call_events.setdefault(call_id,[]).append({"event":"provider_webhook","call_id":call_id,"status":payload.get("status"),"timestamp":now()}); return {"received":True}


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


@app.delete("/api/v1/sessions/{session_id}", status_code=204, response_class=Response, response_model=None)
def delete_session(session_id: str, _: None = Depends(require_api_key)) -> Response:
    if session_id not in service.sessions: raise HTTPException(404, "session not found")
    service.delete_session(session_id)
    return Response(status_code=204)


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


@app.websocket("/stream")
async def neural_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    if inference_queue is None:
        await websocket.send_json({"error":"trained model runtime is not configured"}); await websocket.close(code=1013); return
    sample_rate=int(websocket.headers.get("x-sample-rate",settings.default_sample_rate)); windows=SlidingWindowGenerator(settings.default_sample_rate*3,settings.default_sample_rate)
    try:
        while True:
            message=await websocket.receive();
            if message.get("type")=="websocket.disconnect": break
            if message.get("bytes") is not None: waveform,_=decode_pcm16(message["bytes"],sample_rate,settings.default_sample_rate)
            else: waveform=np.asarray(json.loads(message.get("text","[]")),dtype=np.float32)
            for window in windows.push(waveform):
                if is_speech(window):
                    result=await inference_queue.submit(window); await websocket.send_json({"score":None if settings.hide_raw_scores else result["score"],"tier":result["tier"],"model_version":result["model_version"],"latency_ms":result["latency_ms"]})
    except WebSocketDisconnect: pass


@app.websocket("/api/v1/telephony/dotvoice/media")
async def dotvoice_media(websocket: WebSocket) -> None:
    """Provider media boundary; accepts provider frames and feeds the detector queue.

    DotVoice payload formats can vary by account/API version, so this adapter accepts
    binary PCM16 frames and JSON frames containing base64 PCM16 audio. It never sends
    arbitrary RTP packets or alters the provider call audio.
    """
    await websocket.accept(); call_id=websocket.query_params.get("call_id","unknown"); windows=SlidingWindowGenerator(settings.default_sample_rate*3,settings.default_sample_rate)
    try:
        while True:
            message=await websocket.receive()
            if message.get("type")=="websocket.disconnect": break
            sample_rate=settings.default_sample_rate; payload=message.get("bytes")
            if payload is None:
                data=json.loads(message.get("text","{}")); call_id=str(data.get("call_id",call_id)); sample_rate=int(data.get("sample_rate",sample_rate)); encoded=data.get("audio_base64") or data.get("audio")
                if isinstance(encoded,str): payload=base64.b64decode(encoded)
                elif isinstance(encoded,list): payload=np.asarray(encoded,dtype=np.float32); sample_rate=settings.default_sample_rate
            if isinstance(payload,bytes): waveform,_=decode_pcm16(payload,sample_rate,settings.default_sample_rate)
            else: waveform=np.asarray(payload,dtype=np.float32)
            for window in windows.push(waveform):
                if not is_speech(window): continue
                if inference_queue is None: continue
                result=await inference_queue.submit(window); call_events.setdefault(call_id,[]).append({"event":"risk_update","call_id":call_id,"status":"streaming","risk_tier":result["tier"],"score":None if settings.hide_raw_scores else result["score"],"timestamp":now()})
    except WebSocketDisconnect: pass
