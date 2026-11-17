from __future__ import annotations

import json
from pathlib import Path


class ThresholdPolicy:
    def __init__(self, values: dict[str,float]): self.values=values
    @classmethod
    def from_json(cls,path: str | Path): return cls(json.loads(Path(path).read_text(encoding="utf-8")))
    def tier(self, score: float) -> str:
        if score >= self.values["play_warning"]: return "play_warning"
        if score >= self.values["silent_flag"]: return "silent_flag"
        return "no_action"
