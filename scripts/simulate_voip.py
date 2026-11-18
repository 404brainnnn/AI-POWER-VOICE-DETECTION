from __future__ import annotations

import argparse, json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from app.services import DetectionService
from voip.impairments import ImpairmentConfig
from voip.simulator import run_simulation

parser=argparse.ArgumentParser(description="Offline RTP-like VoIP stream simulator; never sends network packets"); parser.add_argument("--mode",choices=["human","synthetic"],default="synthetic"); parser.add_argument("--wav"); parser.add_argument("--seconds",type=float,default=8); parser.add_argument("--sample-rate",type=int,default=16000); parser.add_argument("--frame-ms",type=int,choices=[10,20,30],default=20); parser.add_argument("--packet-loss",type=float,default=0.0); parser.add_argument("--jitter-ms",type=float,default=0.0); parser.add_argument("--reorder-rate",type=float,default=0.0); parser.add_argument("--snr-db",type=float); parser.add_argument("--bandwidth-hz",type=float); parser.add_argument("--gain",type=float,default=1.0); parser.add_argument("--checkpoint"); parser.add_argument("--model-config",default="configs/model.yaml"); parser.add_argument("--thresholds",default="artifacts/calibration/thresholds.json"); parser.add_argument("--device",default="cuda"); parser.add_argument("--realtime",action="store_true"); parser.add_argument("--output",default="artifacts/voip_simulation.json"); args=parser.parse_args()
if not 0<=args.packet_loss<=1 or not 0<=args.reorder_rate<=1: raise SystemExit("packet loss and reorder rate must be between 0 and 1")
service=DetectionService(args.sample_rate)
if args.checkpoint:
    from models.config import load_yaml_config
    from inference.runtime import NeuralRuntime
    service.detector=NeuralRuntime.from_checkpoint(load_yaml_config(args.model_config),args.checkpoint,args.thresholds,args.device,args.sample_rate)
result=run_simulation(service,args.mode,args.seconds,args.sample_rate,args.wav,ImpairmentConfig(args.packet_loss,args.jitter_ms,args.reorder_rate,args.snr_db,args.bandwidth_hz,args.gain),args.frame_ms,args.realtime); Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result.as_dict(),indent=2),encoding="utf-8"); print(json.dumps(result.as_dict(),indent=2))
