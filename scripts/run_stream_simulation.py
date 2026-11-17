import argparse
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from app.services import DetectionService

parser = argparse.ArgumentParser(); parser.add_argument("--seconds", type=float, default=6); parser.add_argument("--sample-rate", type=int, default=16000)
args = parser.parse_args(); service = DetectionService(args.sample_rate); session = service.create_session(args.sample_rate)
frame = np.sin(2 * np.pi * 220 * np.arange(args.sample_rate // 50) / args.sample_rate).astype(np.float32)
for _ in range(int(args.seconds * 50)):
    for result in service.analyze(session, frame): print(result["decision"])
    time.sleep(0.02)
