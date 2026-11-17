import argparse, json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import classification_metrics

parser=argparse.ArgumentParser(); parser.add_argument("--labels", required=True); parser.add_argument("--scores", required=True); parser.add_argument("--output", default="reports/metrics.json"); args=parser.parse_args()
labels=[int(x) for x in args.labels.split(",")]; scores=[float(x) for x in args.scores.split(",")]
result=classification_metrics(labels,scores); Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
