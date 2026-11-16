from __future__ import annotations

from pathlib import Path


def load_yaml_config(path: str | Path = "configs/model.yaml") -> dict:
    try:
        import yaml
    except ImportError as exc: raise RuntimeError("install PyYAML from requirements-ml.txt") from exc
    with Path(path).open(encoding="utf-8") as handle: return yaml.safe_load(handle)
