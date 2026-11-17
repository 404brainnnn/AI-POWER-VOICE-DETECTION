from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_optimizer(model, config):
    import torch
    groups=model.parameter_groups() if hasattr(model,"parameter_groups") else [{"params":model.parameters(),"lr":config.learning_rate}]
    return torch.optim.AdamW(groups,weight_decay=config.weight_decay)


def train_model(model, train_loader, validation_loader, config, device: str = "cuda", mlflow_run=None) -> dict:
    import torch
    seed_everything(config.seed); device=torch.device(device if device=="cpu" or torch.cuda.is_available() else "cpu"); model.to(device); optimizer=make_optimizer(model,config); total_steps=max(1,config.epochs*max(1,len(train_loader)//config.gradient_accumulation_steps)); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,total_steps)); scaler=torch.amp.GradScaler("cuda",enabled=config.amp in {"fp16","bf16"} and device.type=="cuda"); best_eer=float("inf"); patience=0; history=[]; start_epoch=0
    if config.resume_from: start_epoch=load_checkpoint(config.resume_from,model,optimizer,scheduler,scaler,device)+1
    counts=[0,0]
    for _, labels, _ in train_loader:
        counts[0]+=int((labels==0).sum()); counts[1]+=int((labels==1).sum())
    from .losses import OCSoftmaxLoss, weighted_cross_entropy
    weights, ce_loss=weighted_cross_entropy(counts); loss_fn=OCSoftmaxLoss() if config.loss=="oc_softmax" else ce_loss
    for epoch in range(start_epoch,config.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); train_loss=0.0
        for step,(waveforms,labels,_) in enumerate(train_loader):
            waveforms,labels=waveforms.to(device),labels.to(device)
            autocast_dtype=torch.bfloat16 if config.amp=="bf16" else torch.float16
            with torch.autocast(device_type=device.type,dtype=autocast_dtype,enabled=device.type=="cuda" and config.amp in {"fp16","bf16"}): logits,_=model(waveforms); loss=loss_fn(logits,labels)/config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (step+1)%config.gradient_accumulation_steps==0: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip_norm); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            train_loss += float(loss.detach())
        metrics=evaluate_model(model,validation_loader,device); metrics["train_loss"]=train_loss/max(1,len(train_loader)); metrics["epoch"]=epoch; history.append(metrics)
        save_checkpoint(config.checkpoint_dir+"/last.pt",model,optimizer,scheduler,scaler,epoch,metrics)
        if mlflow_run: mlflow_run.log_metrics({key:value for key,value in metrics.items() if isinstance(value,(float,int))},step=epoch)
        if metrics["eer"]<best_eer: best_eer=metrics["eer"]; patience=0; save_checkpoint(config.checkpoint_dir+"/best.pt",model,optimizer,scheduler,scaler,epoch,metrics)
        else: patience+=1
        if patience>=config.early_stopping_patience: break
    Path(config.output_dir).mkdir(parents=True,exist_ok=True); Path(config.output_dir,"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8"); return {"best_eer":best_eer,"history":history,"device":str(device)}


def evaluate_model(model, loader, device):
    import torch
    model.eval(); scores=[]; labels=[]
    with torch.no_grad():
        for waveforms,target,_ in loader:
            logits,_=model(waveforms.to(device)); scores.extend(torch.softmax(logits,dim=1)[:,1].cpu().numpy()); labels.extend(target.numpy())
    from .evaluation import detection_metrics
    return detection_metrics(np.asarray(scores),np.asarray(labels))


def save_checkpoint(path,model,optimizer,scheduler,scaler,epoch,metrics):
    import torch
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"metrics":metrics},path)


def load_checkpoint(path,model,optimizer=None,scheduler=None,scaler=None,device="cpu"):
    import torch
    checkpoint=torch.load(path,map_location=device); model.load_state_dict(checkpoint["model"])
    if optimizer: optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler: scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler: scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"])
