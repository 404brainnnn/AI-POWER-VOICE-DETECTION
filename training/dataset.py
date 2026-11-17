from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np


def read_manifest(path: str | Path, resolve_relative_to_csv: bool = False) -> list[dict[str, str | int]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows: raise ValueError(f"manifest is empty: {path}")
    columns={str(key).strip().lower():key for key in rows[0] if key is not None}
    if not {"location", "label"}.issubset(columns): raise ValueError(f"manifest must contain location,label columns; found {list(rows[0])}")
    result=[]
    for row in rows:
        location=str(row[columns["location"]]).strip(); label=int(row[columns["label"]])
        if not location: raise ValueError("location cannot be empty")
        if label not in (0,1): raise ValueError("label must be 0 human or 1 AI")
        if resolve_relative_to_csv and not Path(location).is_absolute(): location=str((Path(path).parent/location).resolve())
        result.append({"location": location, "label": label, **{key:value for key,value in row.items() if key not in {"location","label"}}})
    return result


def _load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    try:
        import torchaudio
        waveform, rate = torchaudio.load(str(path))
        values=waveform.mean(dim=0).numpy()
        if rate != sample_rate: values=torchaudio.functional.resample(waveform.mean(dim=0),rate,sample_rate).numpy()
        return values.astype(np.float32)
    except Exception:
        from audio.ingestion import read_pcm_wav
        return read_pcm_wav(path, sample_rate)[0]


def fixed_length_waveform(path: str | Path, sample_rate: int, duration_seconds: float, loader: Callable[[str | Path,int],np.ndarray]=_load_audio) -> np.ndarray:
    values=np.asarray(loader(path,sample_rate),dtype=np.float32).reshape(-1); target=round(sample_rate*duration_seconds)
    if values.size >= target: return values[:target]
    return np.pad(values,(0,target-values.size)).astype(np.float32)


def make_pytorch_dataset(rows: list[dict], sample_rate: int, duration_seconds: float):
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError as exc: raise RuntimeError("install PyTorch from requirements-ml.txt") from exc
    class AudioDataset(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self,index):
            row=rows[index]; waveform=fixed_length_waveform(row["location"],sample_rate,duration_seconds); return torch.from_numpy(waveform), torch.tensor(int(row["label"]),dtype=torch.long), row
    return AudioDataset()


def stratified_split(rows: list[dict], validation_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42) -> dict[str, list[dict]]:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1: raise ValueError("invalid split fractions")
    import random
    grouped={0:[],1:[]}
    for row in rows: grouped[int(row["label"])].append(row)
    rng=random.Random(seed); result={"train":[],"validation":[],"test":[]}
    for label_rows in grouped.values():
        rng.shuffle(label_rows); test_count=round(len(label_rows)*test_fraction); validation_count=round(len(label_rows)*validation_fraction)
        result["test"].extend(label_rows[:test_count]); result["validation"].extend(label_rows[test_count:test_count+validation_count]); result["train"].extend(label_rows[test_count+validation_count:])
    for values in result.values(): values.sort(key=lambda row: str(row["location"]))
    if not result["train"] or not result["validation"]: raise ValueError("dataset is too small for train/validation split")
    return result
