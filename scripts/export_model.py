from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.config import load_yaml_config
from training.predict import load_trained_model

parser=argparse.ArgumentParser(); parser.add_argument("--model-config",default="configs/model.yaml"); parser.add_argument("--checkpoint",required=True); parser.add_argument("--output",required=True); parser.add_argument("--format",choices=["torchscript","onnx"],default="torchscript"); parser.add_argument("--sample-rate",type=int,default=16000); args=parser.parse_args(); model,device=load_trained_model(load_yaml_config(args.model_config),args.checkpoint,"cuda"); example=torch.zeros(1,args.sample_rate*3,device=device); Path(args.output).parent.mkdir(parents=True,exist_ok=True)
if args.format=="torchscript": torch.jit.trace(model,example).save(args.output)
else: torch.onnx.export(model,example,args.output,input_names=["waveform"],output_names=["logits","embedding"],dynamic_axes={"waveform":{1:"samples"},"logits":{0:"batch"},"embedding":{0:"batch"}},opset_version=17)
print(f"exported {args.format} model to {args.output}")
