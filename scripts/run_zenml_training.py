import argparse
from pipelines.zenml_training import build_training_pipeline

parser=argparse.ArgumentParser(); parser.add_argument("--csv",required=True); parser.add_argument("--config",default="configs/train.yaml"); args=parser.parse_args(); pipeline=build_training_pipeline(); pipeline(args.csv,args.config); print("ZenML training pipeline submitted")
