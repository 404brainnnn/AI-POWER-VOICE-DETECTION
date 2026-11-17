from __future__ import annotations

import numpy as np


def load_trained_model(model_config: dict, checkpoint_path: str, device: str = "cuda"):
    import torch
    from models.factory import build_detector
    model=build_detector(model_config); checkpoint=torch.load(checkpoint_path,map_location="cpu"); model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint); target=torch.device(device if device=="cpu" or torch.cuda.is_available() else "cpu"); return model.to(target).eval(),target


def collect_logits(model, loader, device):
    import torch
    logits=[]; labels=[]
    with torch.no_grad():
        for waveforms,target,_ in loader:
            output,_=model(waveforms.to(device)); logits.append(output.cpu().numpy()); labels.append(target.numpy())
    return np.concatenate(logits),np.concatenate(labels)
