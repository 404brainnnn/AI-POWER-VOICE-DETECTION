from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT

import numpy as np

from models.config import load_yaml_config
from models.factory import build_detector
from training.config import load_config
from training.dataset import make_pytorch_dataset, read_manifest, stratified_split
from training.loop import train_model


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/train.yaml"); parser.add_argument("--device",default="cuda"); parser.add_argument("--mlflow",action="store_true"); args=parser.parse_args(); cfg=load_config(args.config); rows=read_manifest(cfg.data.csv_path)
    import torch
    from torch.utils.data import DataLoader
    splits=stratified_split(rows, seed=cfg.train.seed); train_set=make_pytorch_dataset(splits["train"],cfg.data.sample_rate,cfg.data.duration_seconds); val_set=make_pytorch_dataset(splits["validation"],cfg.data.sample_rate,cfg.data.duration_seconds); train_loader=DataLoader(train_set,batch_size=cfg.data.batch_size,shuffle=True,num_workers=cfg.data.num_workers); val_loader=DataLoader(val_set,batch_size=cfg.data.batch_size,shuffle=False,num_workers=cfg.data.num_workers)
    model_config=load_yaml_config("configs/model.yaml"); model_config["model"]["name"]=cfg.train.model; model_config["model"]["backbone"]=cfg.train.backbone; model=build_detector(model_config)
    if args.mlflow:
        import mlflow
        from training.mlflow import ActiveRunLogger
        mlflow.set_tracking_uri(cfg.train.mlflow_tracking_uri or "http://localhost:5000"); mlflow.set_experiment(cfg.train.mlflow_experiment)
        with mlflow.start_run() as run:
            mlflow.log_params({"model":cfg.train.model,"backbone":cfg.train.backbone,"seed":cfg.train.seed,"loss":cfg.train.loss,"amp":cfg.train.amp,"csv_path":cfg.data.csv_path}); result=train_model(model,train_loader,val_loader,cfg.train,args.device,ActiveRunLogger()); mlflow.log_artifact(str(Path(cfg.train.checkpoint_dir)/"best.pt"),artifact_path="checkpoints"); print({"run_id":run.info.run_id,**result})
    else: result=train_model(model,train_loader,val_loader,cfg.train,args.device); print(result)


if __name__ == "__main__": main()
