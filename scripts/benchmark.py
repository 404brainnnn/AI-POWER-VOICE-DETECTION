import argparse, json, time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dsp.features import extract_features

parser=argparse.ArgumentParser(); parser.add_argument("--iterations",type=int,default=10); args=parser.parse_args(); audio=np.zeros(16000,dtype=np.float32); values=[]
for _ in range(args.iterations):
    start=time.perf_counter(); extract_features(audio,16000); values.append((time.perf_counter()-start)*1000)
values.sort(); result={"iterations":args.iterations,"p50_ms":values[int(.5*(len(values)-1))],"p95_ms":values[int(.95*(len(values)-1))],"p99_ms":values[int(.99*(len(values)-1))],"observation_delay_ms":3000.0}; print(json.dumps(result,indent=2))
