# API

Run `uvicorn app.main:app --reload`, then open `/docs`.

```bash
curl -X POST 'http://localhost:8000/api/v1/sessions?sample_rate=16000'
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/detection/analyze -H 'content-type: application/json' -d '{"sample_rate":16000,"audio":[0.01,0.02,0.0]}'
```

WebSocket: connect to `ws://localhost:8000/api/v1/stream/{session_id}` and send JSON frames such as `{"sample_rate":16000,"audio":[...]}`. A response is emitted only after a speech-positive complete window.

When no trained DSP model is configured, `dsp_score`, `fused_score`, and model versions are `null`/empty; the response explicitly reports `insufficient_evidence`. The API never fabricates a score.

Authorized DotVoice demo calls use `POST /api/v1/calls` with `{ "to": "+...", "mode": "human" | "synthetic", "consent_confirmed": true }`. DotVoice credentials stay on the backend. The provider webhook is `/api/v1/telephony/dotvoice/webhook` and the media boundary is `wss://<public-host>/api/v1/telephony/dotvoice/media`.
