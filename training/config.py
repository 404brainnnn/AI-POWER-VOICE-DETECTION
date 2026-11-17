from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    csv_path: str = "data/manifest.csv"
    train_csv_path: str | None = None
    validation_csv_path: str | None = None
    test_csv_path: str | None = None
    sample_rate: int = 16_000
    duration_seconds: float = 3.0
    num_workers: int = 2
    batch_size: int = 4


@dataclass
class TrainConfig:
    seed: int = 42
    model: str = "baseline_cnn"
    backbone: str = "lcnn"
    epochs: int = 20
    learning_rate: float = 2e-4
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    grad_clip_norm: float = 1.0
    amp: str = "bf16"
    loss: str = "weighted_cross_entropy"
    early_stopping_patience: int = 5
    checkpoint_dir: str = "artifacts/checkpoints"
    output_dir: str = "artifacts/run"
    resume_from: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "voice-deepfake"
    verbose: bool = True
    log_interval: int = 10


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    try:
        import yaml
    except ImportError as exc: raise RuntimeError("install PyYAML from requirements-ml.txt") from exc
    with Path(path).open(encoding="utf-8") as handle: raw = yaml.safe_load(handle) or {}
    return ExperimentConfig(DataConfig(**raw.get("data", {})), TrainConfig(**raw.get("train", {})))
