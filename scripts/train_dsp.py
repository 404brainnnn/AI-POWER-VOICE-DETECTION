"""Train a DSP detector from a CSV with audio_path and label columns."""
import argparse, csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio.ingestion import read_pcm_wav
from dsp.features import extract_features
from models.dsp_detector import DSPDetector

parser = argparse.ArgumentParser(); parser.add_argument("manifest"); parser.add_argument("output")
args = parser.parse_args(); rows=[]; labels=[]
with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
    for record in csv.DictReader(handle):
        audio, rate = read_pcm_wav(record["audio_path"]); rows.append(extract_features(audio, rate)); labels.append(1 if record["label"] == "deepfake" else 0)
if not rows: raise SystemExit("manifest has no rows")
DSPDetector().fit(rows, labels).save(args.output); print(f"saved trained DSP model to {args.output}")
