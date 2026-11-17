from __future__ import annotations

import argparse

from _bootstrap import ROOT

parser=argparse.ArgumentParser(); parser.add_argument("--run-id",required=True); parser.add_argument("--checkpoint",required=True); parser.add_argument("--model-name",default="voice-deepfake-detector"); parser.add_argument("--thresholds"); parser.add_argument("--stage",choices=["Staging","Production"]); args=parser.parse_args()
from training.mlflow import register_checkpoint
version=register_checkpoint(args.run_id,args.checkpoint,args.model_name,args.thresholds)
if args.stage:
    import mlflow
    mlflow.MlflowClient().transition_model_version_stage(args.model_name,version,args.stage,archive_existing_versions=args.stage=="Production")
print({"model_name":args.model_name,"version":version,"stage":args.stage})
