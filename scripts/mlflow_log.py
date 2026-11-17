import argparse
from _bootstrap import ROOT
from monitoring.tracking import log_run

parser=argparse.ArgumentParser(); parser.add_argument("--experiment", default="voice-clone-dsp"); parser.add_argument("--roc-auc", type=float, required=True); args=parser.parse_args()
try:
    import mlflow
    mlflow.set_experiment(args.experiment)
except ImportError: raise SystemExit("install the optional mlops dependencies")
run_id=log_run({"experiment": args.experiment, "research_result": "user-supplied"}, {"roc_auc": args.roc_auc})
print(run_id or "MLflow unavailable")
