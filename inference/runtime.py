from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from audio.ingestion import read_pcm_wav
from .thresholds import ThresholdPolicy


class NeuralRuntime:
    def __init__(self, model, device, model_version: str, thresholds: ThresholdPolicy, sample_rate: int = 16000): self.model,self.device,self.model_version,self.thresholds,self.sample_rate=model,device,model_version,thresholds,sample_rate
    @classmethod
    def from_checkpoint(cls, model_config: dict, checkpoint: str, thresholds_path: str, device: str = "cuda", sample_rate: int = 16000):
        from training.predict import load_trained_model
        model,target=load_trained_model(model_config,checkpoint,device); policy=ThresholdPolicy.from_json(thresholds_path); runtime=cls(model,target,Path(checkpoint).stem,policy,sample_rate); runtime.warmup(); return runtime
    def warmup(self): self.predict_batch([np.zeros(self.sample_rate*3,dtype=np.float32)])
    def predict_batch(self,waveforms: list[np.ndarray]) -> list[dict]:
        import torch
        started=time.perf_counter(); batch=torch.tensor(np.stack(waveforms),dtype=torch.float32,device=self.device)
        with torch.inference_mode(): logits,_=self.model(batch); scores=torch.softmax(logits,dim=1)[:,1].detach().cpu().numpy()
        latency=(time.perf_counter()-started)*1000
        return [{"score":float(score),"tier":self.thresholds.tier(float(score)),"model_version":self.model_version,"latency_ms":latency} for score in scores]

    def predict_proba(self, waveform: np.ndarray, sample_rate: int):
        from models.base import DetectorOutput
        if sample_rate != self.sample_rate: raise ValueError("runtime sample rate mismatch")
        result=self.predict_batch([waveform])[0]
        return DetectorOutput(result["score"],result["model_version"],True,"neural")
