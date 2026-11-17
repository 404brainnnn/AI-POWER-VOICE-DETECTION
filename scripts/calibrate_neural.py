from __future__ import annotations

import argparse, json
from pathlib import Path

from models.config import load_yaml_config
from training.calibration import calibrated_scores, choose_thresholds, expected_calibration_error, save_thresholds, temperature_scale
from training.config import load_config
from training.dataset import make_pytorch_dataset, read_manifest, stratified_split
from training.predict import collect_logits, load_trained_model

parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/train.yaml"); parser.add_argument("--checkpoint",required=True); parser.add_argument("--csv"); parser.add_argument("--output-dir",default="artifacts/calibration"); parser.add_argument("--device",default="cuda"); args=parser.parse_args(); cfg=load_config(args.config); rows=read_manifest(args.csv or cfg.data.csv_path); validation=stratified_split(rows,seed=cfg.train.seed)["validation"]
import torch
from torch.utils.data import DataLoader
model_config=load_yaml_config("configs/model.yaml"); model_config["model"]["name"]=cfg.train.model; model_config["model"]["backbone"]=cfg.train.backbone; model,device=load_trained_model(model_config,args.checkpoint,args.device); loader=DataLoader(make_pytorch_dataset(validation,cfg.data.sample_rate,cfg.data.duration_seconds),batch_size=cfg.data.batch_size,shuffle=False); logits,labels=collect_logits(model,loader,device); temperature=temperature_scale(logits,labels); scores=calibrated_scores(logits,temperature); thresholds=choose_thresholds(scores,labels); Path(args.output_dir).mkdir(parents=True,exist_ok=True); save_thresholds(Path(args.output_dir)/"thresholds.json",thresholds); Path(args.output_dir,"calibration.json").write_text(json.dumps({"temperature":temperature,"ece":expected_calibration_error(scores,labels),"num_samples":len(labels)},indent=2),encoding="utf-8"); print(json.dumps({"temperature":temperature,"ece":expected_calibration_error(scores,labels),"thresholds":thresholds},indent=2))
