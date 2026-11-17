"""ZenML pipeline wrappers around the real training functions.

ZenML is imported lazily so DSP-only installation remains usable.
"""
from __future__ import annotations


def build_training_pipeline():
    try:
        from zenml import pipeline, step
    except ImportError as exc: raise RuntimeError("install the optional MLOps dependencies") from exc

    @step
    def load_data(csv_path: str) -> dict: 
        from training.dataset import read_manifest
        return {"rows": read_manifest(csv_path)}

    @step
    def train(data: dict, config_path: str) -> dict:
        import torch
        from torch.utils.data import DataLoader
        from models.config import load_yaml_config
        from models.factory import build_detector
        from training.config import load_config
        from training.dataset import make_pytorch_dataset, stratified_split
        from training.loop import train_model
        config=load_config(config_path); splits=stratified_split(data["rows"],seed=config.train.seed); train_loader=DataLoader(make_pytorch_dataset(splits["train"],config.data.sample_rate,config.data.duration_seconds),batch_size=config.data.batch_size,shuffle=True,num_workers=config.data.num_workers); val_loader=DataLoader(make_pytorch_dataset(splits["validation"],config.data.sample_rate,config.data.duration_seconds),batch_size=config.data.batch_size,shuffle=False,num_workers=config.data.num_workers); model_config=load_yaml_config("configs/model.yaml"); model_config["model"]["name"]=config.train.model; model_config["model"]["backbone"]=config.train.backbone; model=build_detector(model_config); result=train_model(model,train_loader,val_loader,config.train,"cuda"); return {"checkpoint":config.train.checkpoint_dir+"/best.pt",**{key:value for key,value in result.items() if isinstance(value,(float,int,str))}}

    @step
    def evaluate(training_result: dict) -> dict: return training_result

    @step
    def calibrate(evaluation_result: dict) -> dict: return evaluation_result

    @step
    def register(calibration_result: dict) -> dict: return {"status": "ready_for_registry", **calibration_result}

    @pipeline
    def training_pipeline(csv_path: str, config_path: str):
        loaded=load_data(csv_path); trained=train(loaded,config_path); evaluated=evaluate(trained); calibrated=calibrate(evaluated); return register(calibrated)
    return training_pipeline
