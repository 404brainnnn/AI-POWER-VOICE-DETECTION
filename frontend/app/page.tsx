"use client";

import { FormEvent, useEffect, useState } from "react";

type Call = { call_id: string; status: string; mode: string; to: string };
type EventItem = { event: string; status?: string; risk_tier?: string; score?: number | null; timestamp: string };

const API = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function Home() {
  const [number, setNumber] = useState("");
  const [mode, setMode] = useState<"human" | "synthetic">("synthetic");
  const [consent, setConsent] = useState(false);
  const [call, setCall] = useState<Call | null>(null);
  const [events, setEvents] = useState<EventItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!call) return;
    const timer = window.setInterval(async () => {
      const response = await fetch(`${API}/api/v1/calls/${call.call_id}/events`);
      if (response.ok) setEvents(await response.json());
    }, 1500);
    return () => window.clearInterval(timer);
  }, [call]);

  async function dial(event: FormEvent) {
    event.preventDefault(); setError(""); setBusy(true);
    try {
      const response = await fetch(`${API}/api/v1/calls`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ to: number, mode, consent_confirmed: consent }) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "Call could not be created");
      setCall(body); setEvents([]);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Call could not be created"); }
    finally { setBusy(false); }
  }

  const latestRisk = [...events].reverse().find((item) => item.event === "risk_update");

  return <main className="shell">
    <nav className="topbar"><div className="brand"><span className="brand-dot" />voiceguard</div><div className="nav-meta"><span className="live-dot" />authorized demo console <span className="env-pill">LOCAL</span></div></nav>
    <section className="hero"><div className="eyebrow">REAL-TIME VOICE SECURITY / 01</div><h1>See the signal<br /><em>behind the voice.</em></h1><p className="lede">Place an authorized test call, choose a reference voice mode, and watch the detector accumulate evidence across the live stream.</p></section>
    <section className="workspace">
      <form className="panel dial-panel" onSubmit={dial}><div className="panel-kicker">01 / START A TEST CALL</div><h2>Dial a number</h2><label className="field-label" htmlFor="phone">Destination number</label><input id="phone" value={number} onChange={(event) => setNumber(event.target.value)} placeholder="+91 98765 43210" required />
        <div className="mode-heading"><span className="field-label">Voice mode</span><span className="mode-note">provider-controlled</span></div><div className="mode-grid"><button type="button" className={mode === "human" ? "mode selected" : "mode"} onClick={() => setMode("human")}><span className="mode-icon">◌</span><span><strong>Human reference</strong><small>Reference speech path</small></span></button><button type="button" className={mode === "synthetic" ? "mode selected synthetic" : "mode synthetic"} onClick={() => setMode("synthetic")}><span className="mode-icon">✦</span><span><strong>Synthetic demo</strong><small>Disclosed AI test path</small></span></button></div>
        <label className="consent"><input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} /><span>I own or have permission to call this number for testing.</span></label><button className="dial-button" disabled={busy || !consent}>{busy ? "Connecting…" : "Start authorized call"}<span>↗</span></button>{error && <p className="error">{error}</p>}<p className="microcopy">API credentials remain server-side. The dashboard never receives the DotVoice key.</p>
      </form>
      <section className="panel signal-panel"><div className="panel-kicker">02 / DETECTION TELEMETRY</div><div className="signal-header"><div><h2>Live signal</h2><p>{call ? `Call ${call.call_id}` : "Waiting for a test call"}</p></div><div className={`status ${call ? "active" : ""}`}><span />{call ? call.status : "idle"}</div></div><div className="risk-card"><div className="risk-label">CURRENT RISK TIER</div><div className="risk-value">{latestRisk?.risk_tier?.replace("_", " ") ?? "—"}</div><div className="risk-score">{latestRisk?.score != null ? `${(latestRisk.score * 100).toFixed(1)}% synthetic probability` : "A score appears after three seconds of speech"}</div></div><div className="timeline">{events.length === 0 ? <div className="empty-state"><div className="pulse-ring" /><p>No signal events yet.</p><small>Start a call to populate the stream.</small></div> : events.slice(-6).reverse().map((item, index) => <div className="event-row" key={`${item.timestamp}-${index}`}><span className="event-time">{new Date(item.timestamp).toLocaleTimeString()}</span><span className={`event-marker ${item.risk_tier ?? ""}`} /><span>{item.event.replaceAll("_", " ")}</span><b>{item.risk_tier?.replace("_", " ") ?? item.status ?? ""}</b></div>)}</div></section>
    </section>
    <footer><span>VOICEGUARD / DOTVOICE ADAPTER</span><span>Human review recommended for every high-risk event.</span></footer>
  </main>;
}
