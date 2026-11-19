"use client";

import { useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const SAMPLE_RATE = 16_000;
const FRAME_MS = 20;
const FRAME_SAMPLES = (SAMPLE_RATE * FRAME_MS) / 1000;
const WINDOW_SAMPLES = SAMPLE_RATE * 3;

type Mode = "human" | "ai";
type Phase = "idle" | "streaming" | "complete" | "error";
type StreamEvent = { id: number; time: string; kind: string; detail: string; tone?: "accent" | "warning" | "ok" };
type ScoreResult = { score: number | null; tier: string; model_version: string; latency_ms: number };

const sleep = (milliseconds: number) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function resampleMono(source: Float32Array, sourceRate: number): Float32Array {
  if (sourceRate === SAMPLE_RATE) return new Float32Array(source);
  const outputLength = Math.round(source.length * SAMPLE_RATE / sourceRate);
  const output = new Float32Array(outputLength);
  for (let i = 0; i < output.length; i += 1) {
    const position = i * sourceRate / SAMPLE_RATE;
    const left = Math.floor(position);
    const right = Math.min(left + 1, source.length - 1);
    const fraction = position - left;
    output[i] = source[left] * (1 - fraction) + source[right] * fraction;
  }
  return output;
}

function pcm16Bytes(samples: Float32Array): ArrayBuffer {
  const output = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) output[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)));
  return output.buffer;
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("ai");
  const [humanFile, setHumanFile] = useState<File | null>(null);
  const [aiFile, setAiFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [score, setScore] = useState<ScoreResult | null>(null);
  const [packetCount, setPacketCount] = useState(0);
  const [windowCount, setWindowCount] = useState(0);
  const [error, setError] = useState("");
  const runId = useRef(0);

  function addEvent(kind: string, detail: string, tone?: StreamEvent["tone"]) {
    setEvents((current) => [{ id: Date.now() + current.length, time: new Date().toLocaleTimeString(), kind, detail, tone }, ...current].slice(0, 18));
  }

  async function executeSimulation() {
    const thisRun = runId.current + 1;
    runId.current = thisRun;
    setPhase("streaming"); setError(""); setEvents([]); setScore(null); setPacketCount(0); setWindowCount(0);
    const selectedFile = mode === "ai" ? aiFile : humanFile;
    if (!selectedFile) { setPhase("error"); setError(`Upload a ${mode === "ai" ? "AI" : "human"} voice file first`); return; }
    try {
      const health = await fetch(`${API}/healthz`);
      if (!health.ok) throw new Error("Detector backend is not ready");
      const audioContext = new AudioContext();
      const decoded = await audioContext.decodeAudioData(await selectedFile.arrayBuffer());
      const source = decoded.getChannelData(0);
      const waveform = resampleMono(source, decoded.sampleRate);
      await audioContext.close();
      if (waveform.length < WINDOW_SAMPLES) throw new Error("Audio must contain at least 3 seconds of speech");
      addEvent("simulation_started", `${mode === "ai" ? "AI" : "human"} upload: ${selectedFile.name}`, "ok");
      const frames: Float32Array[] = [];
      const totalFrames = Math.ceil(waveform.length / FRAME_SAMPLES);
      for (let frameNumber = 0; frameNumber < totalFrames; frameNumber += 1) {
        if (runId.current !== thisRun) return;
        const frame = new Float32Array(FRAME_SAMPLES);
        frame.set(waveform.subarray(frameNumber * FRAME_SAMPLES, Math.min((frameNumber + 1) * FRAME_SAMPLES, waveform.length)));
        frames.push(frame);
        const packetNumber = frameNumber + 1;
        setPacketCount(packetNumber);
        if (packetNumber % 10 === 0) addEvent("packet_received", `RTP-like packet ${packetNumber} delivered`, "ok");
        if (frames.length >= WINDOW_SAMPLES / FRAME_SAMPLES && packetNumber % (1000 / FRAME_MS) === 0) {
          const window = new Float32Array(WINDOW_SAMPLES);
          frames.slice(-WINDOW_SAMPLES / FRAME_SAMPLES).forEach((frame, index) => window.set(frame, index * FRAME_SAMPLES));
          setWindowCount((current) => current + 1);
          addEvent("window_ready", "3.0 s speech window sent to detector");
          const response = await fetch(`${API}/score`, { method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Sample-Rate": String(SAMPLE_RATE) }, body: pcm16Bytes(window) });
          const body = await response.json();
          if (!response.ok) throw new Error(body.detail ?? "Model inference failed");
          const result = body as ScoreResult;
          setScore(result);
          addEvent("model_result", `${result.tier} · ${result.score == null ? "score hidden" : `${(result.score * 100).toFixed(1)}% spoof probability`} · ${result.latency_ms.toFixed(1)} ms`, "accent");
          addEvent("packet_inserted", "simulated control packet inserted after decision", "warning");
        }
        await sleep(FRAME_MS);
      }
      if (runId.current === thisRun) { setPhase("complete"); addEvent("simulation_complete", "offline VoIP stream finished", "ok"); }
    } catch (reason) {
      if (runId.current !== thisRun) return;
      setPhase("error"); setError(reason instanceof TypeError ? `Cannot reach detector at ${API}. Check the SSH port-forward.` : reason instanceof Error ? reason.message : "Simulation failed"); addEvent("simulation_error", "detector request failed", "warning");
    }
  }

  function stopSimulation() { runId.current += 1; setPhase("idle"); addEvent("simulation_stopped", "stream stopped by operator", "warning"); }
  const riskTier = score?.tier?.replaceAll("_", " ") ?? "waiting";

  return <main className="shell">
    <nav className="topbar"><div className="brand"><span className="brand-dot" />voiceguard</div><div className="nav-meta"><span className="live-dot" />offline voip lab <span className="env-pill">GPU MODEL</span></div></nav>
    <section className="hero"><div className="eyebrow">REAL-TIME VOICE SECURITY / SIMULATION</div><h1>See the signal<br /><em>inside the stream.</em></h1><p className="lede">Run a disclosed Human or AI voice fixture through an RTP-like packet stream and watch the trained detector score rolling three-second windows.</p></section>
    <section className="workspace">
      <section className="panel dial-panel"><div className="panel-kicker">01 / CONFIGURE THE STREAM</div><h2>Choose a voice path</h2><div className="field-label">Input recording</div><div className="mode-grid"><button type="button" className={mode === "human" ? "mode selected" : "mode"} onClick={() => setMode("human")} disabled={phase === "streaming"}><span className="mode-icon">◌</span><span><strong>Human voice</strong><small>{humanFile?.name ?? "Select reference audio"}</small></span></button><button type="button" className={mode === "ai" ? "mode selected synthetic" : "mode synthetic"} onClick={() => setMode("ai")} disabled={phase === "streaming"}><span className="mode-icon">✦</span><span><strong>AI voice</strong><small>{aiFile?.name ?? "Select synthetic audio"}</small></span></button></div><label className="upload-field">Upload human voice<input type="file" accept="audio/*" onChange={(event) => setHumanFile(event.target.files?.[0] ?? null)} disabled={phase === "streaming"} /></label><label className="upload-field">Upload AI voice<input type="file" accept="audio/*" onChange={(event) => setAiFile(event.target.files?.[0] ?? null)} disabled={phase === "streaming"} /></label><div className="sim-specs"><div><span>PACKET SIZE</span><b>20 ms</b></div><div><span>WINDOW</span><b>3.0 s</b></div><div><span>HOP</span><b>1.0 s</b></div></div><button className="dial-button" onClick={executeSimulation} disabled={phase === "streaming"}>{phase === "streaming" ? "Streaming…" : "Execute simulation"}<span>↗</span></button>{phase === "streaming" && <button className="stop-button" onClick={stopSimulation}>Stop stream</button>}{error && <p className="error">{error}</p>}<p className="microcopy">Audio is decoded in memory, packetized locally, and sent to the GPU detector as rolling PCM16 windows. No phone call is made.</p></section>
      <section className="panel signal-panel"><div className="panel-kicker">02 / LIVE DETECTION TELEMETRY</div><div className="signal-header"><div><h2>Packet stream</h2><p>{phase === "idle" ? "Ready for execution" : `${mode === "ai" ? "AI" : "Human"} fixture · ${phase}`}</p></div><div className={`status ${phase === "streaming" ? "active" : ""}`}><span />{phase}</div></div><div className="risk-card"><div className="risk-label">MODEL DECISION</div><div className="risk-value">{riskTier}</div><div className="risk-score">{score?.score == null ? "Score appears after the first complete window" : `${(score.score * 100).toFixed(2)}% calibrated spoof probability · ${score.model_version}`}</div></div><div className="stream-stats"><div><span>PACKETS</span><b>{packetCount}</b></div><div><span>WINDOWS SCORED</span><b>{windowCount}</b></div><div><span>INFERENCE</span><b>{score ? `${score.latency_ms.toFixed(1)} ms` : "—"}</b></div></div><div className="timeline">{events.length === 0 ? <div className="empty-state"><div className="pulse-ring" /><p>No stream events yet.</p><small>Choose a voice path and execute the simulation.</small></div> : events.map((item) => <div className="event-row" key={item.id}><span className="event-time">{item.time}</span><span className={`event-marker ${item.tone ?? ""}`} /><span>{item.kind.replaceAll("_", " ")}</span><b>{item.detail}</b></div>)}</div></section>
    </section>
    <footer><span>VOICEGUARD / OFFLINE RTP-LIKE SIMULATION</span><span>Packets are simulated; model output is produced by the configured GPU runtime.</span></footer>
  </main>;
}
