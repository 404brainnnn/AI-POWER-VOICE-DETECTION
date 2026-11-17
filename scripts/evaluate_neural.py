from __future__ import annotations

import argparse, json
from pathlib import Path

from models.config import load_yaml_config
from training.config import load_config
from training.dataset import make_pytorch_dataset, read_manifest, stratified_split
from training.evaluation import detection_metrics
from training.predict import collect_logits, load_trained_model
from training.calibration import calibrated_scores

parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/train.yaml"); parser.add_argument("--checkpoint",required=True); parser.add_argument("--csv"); parser.add_argument("--calibration"); parser.add_argument("--device",default="cuda"); parser.add_argument("--output",default="artifacts/evaluation.json"); args=parser.parse_args(); cfg=load_config(args.config); rows=read_manifest(args.csv or cfg.data.csv_path); test=stratified_split(rows,seed=cfg.train.seed)["test"]
import torch
from torch.utils.data import DataLoader
model_config=load_yaml_config("configs/model.yaml"); model_config["model"]["name"]=cfg.train.model; model_config["model"]["backbone"]=cfg.train.backbone; model,device=load_trained_model(model_config,args.checkpoint,args.device); loader=DataLoader(make_pytorch_dataset(test,cfg.data.sample_rate,cfg.data.duration_seconds),batch_size=cfg.data.batch_size,shuffle=False); logits,labels=collect_logits(model,loader,device); temperature=1.0
if args.calibration: temperature=json.loads(Path(args.calibration).read_text(encoding="utf-8"))["temperature"]
scores=calibrated_scores(logits,temperature); result=detection_metrics(scores,labels); Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
