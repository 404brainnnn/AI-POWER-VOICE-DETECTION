from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    version: str
    kind: str
    calibrated: bool
    status: str


class ModelRegistry:
    def __init__(self): self._models: dict[str, ModelMetadata] = {}
    def register(self, metadata: ModelMetadata) -> None:
        if metadata.status == "production" and not metadata.calibrated: raise ValueError("production models require calibration metadata")
        self._models[metadata.name] = metadata
    def list(self) -> list[ModelMetadata]: return sorted(self._models.values(), key=lambda item: item.name)
    def get(self, name: str) -> ModelMetadata: return self._models[name]
