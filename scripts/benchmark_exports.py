from __future__ import annotations

import argparse, json, time
from pathlib import Path

from _bootstrap import ROOT

import numpy as np

parser=argparse.ArgumentParser(); parser.add_argument("--torchscript"); parser.add_argument("--onnx"); parser.add_argument("--iterations",type=int,default=20); parser.add_argument("--sample-rate",type=int,default=16000); parser.add_argument("--output",default="artifacts/export_benchmark.json"); args=parser.parse_args()

def benchmark(kind,path):
    import torch
    waveform=torch.zeros(1,args.sample_rate*3)
    if kind=="torchscript": model=torch.jit.load(path,map_location="cpu").eval(); runner=lambda:model(waveform)
    else:
        import onnxruntime as ort
        session=ort.InferenceSession(path,providers=["CPUExecutionProvider"]); name=session.get_inputs()[0].name; runner=lambda:session.run(None,{name:waveform.numpy()})
    for _ in range(3): runner()
    values=[]
    for _ in range(args.iterations): start=time.perf_counter(); runner(); values.append((time.perf_counter()-start)*1000)
    values.sort(); return {"format":kind,"artifact_bytes":Path(path).stat().st_size,"p50_ms":values[int(.5*(len(values)-1))],"p95_ms":values[int(.95*(len(values)-1))],"p99_ms":values[int(.99*(len(values)-1))],"eer":None,"note":"EER requires scores from a held-out labeled evaluation set; no research EER is inferred from random inputs."}
result=[]
if args.torchscript: result.append(benchmark("torchscript",args.torchscript))
if args.onnx: result.append(benchmark("onnx",args.onnx))
if not result: raise SystemExit("provide --torchscript and/or --onnx")
Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
