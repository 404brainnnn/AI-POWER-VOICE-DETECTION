from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time

import numpy as np

from audio.ingestion import read_pcm_wav
from .impairments import ImpairmentConfig, impair_audio, reorder_packets
from .rtp import RTPPacket


@dataclass
class SimulationResult:
    mode: str
    input_seconds: float
    packets_created: int
    packets_delivered: int
    packets_dropped: int
    packets_reordered: int
    detector_windows: int
    decisions: list[dict]
    elapsed_seconds: float
    def as_dict(self): return asdict(self)


def fixture_waveform(mode: str, sample_rate: int, seconds: float) -> np.ndarray:
    """Generate deterministic lab audio; it is not a voice-cloning model or result."""
    count=int(sample_rate*seconds); t=np.arange(count)/sample_rate; base=220 if mode=="human" else 330; signal=.18*np.sin(2*np.pi*base*t)+.05*np.sin(2*np.pi*(base*2.01)*t); envelope=.5+.5*np.sin(2*np.pi*2*t); return (signal*envelope).astype(np.float32)


def packetize(audio: np.ndarray, sample_rate: int = 16_000, frame_ms: int = 20, start_sequence: int = 0) -> list[RTPPacket]:
    frame_samples=round(sample_rate*frame_ms/1000); packets=[]
    for index,start in enumerate(range(0,len(audio),frame_samples)):
        frame=np.clip(audio[start:start+frame_samples],-1,1)
        if len(frame)<frame_samples: frame=np.pad(frame,(0,frame_samples-len(frame)))
        packets.append(RTPPacket(start_sequence+index,start,((frame*32767).astype("<i2")).tobytes(),sample_rate,11,index==0))
    return packets


def run_simulation(service, mode: str = "synthetic", seconds: float = 8.0, sample_rate: int = 16_000, wav_path: str | None = None, impairment: ImpairmentConfig | None = None, frame_ms: int = 20, realtime: bool = False) -> SimulationResult:
    started=time.perf_counter(); impairment=impairment or ImpairmentConfig(); audio=read_pcm_wav(wav_path,sample_rate)[0] if wav_path else fixture_waveform(mode,sample_rate,seconds); audio=impair_audio(audio,sample_rate,impairment); packets=packetize(audio,sample_rate,frame_ms); rng=np.random.default_rng(impairment.seed); delivered=[]
    for packet in packets:
        if rng.random()<impairment.packet_loss_rate: continue
        delivered.append(packet)
        if realtime: time.sleep(max(0.0,frame_ms/1000 + float(rng.normal(0,impairment.jitter_ms/1000))))
    reordered=reorder_packets(delivered,impairment.reorder_rate,impairment.seed); session=service.create_session(sample_rate); decisions=[]
    for packet in reordered:
        waveform=np.frombuffer(packet.payload,dtype="<i2").astype(np.float32)/32767.0
        decisions.extend(service.analyze(session,waveform))
    return SimulationResult(mode,len(audio)/sample_rate,len(packets),len(delivered),len(packets)-len(delivered),sum(a.sequence>b.sequence for a,b in zip(reordered,reordered[1:])),len(decisions),[{"tier":item["decision"].tier,"score":item["dsp_score"],"latency_ms":item["latency_ms"]} for item in decisions],time.perf_counter()-started)
