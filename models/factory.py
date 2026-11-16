from __future__ import annotations

from typing import Any

from .neural.baseline_cnn import BaselineCNN, BaselineCNNConfig
from .neural.xlsr_aasist import XLSRAASIST, XLSRAASISTConfig


def build_detector(config: dict[str, Any]):
    """One-line model switch: ``model.name`` is ``baseline_cnn`` or ``xlsr_aasist``."""
    name = config.get("model", {}).get("name", "baseline_cnn")
    if name == "baseline_cnn": return BaselineCNN(BaselineCNNConfig(**config.get("model", {}).get("baseline", {})), config.get("model", {}).get("backbone", "lcnn"))
    if name == "xlsr_aasist": return XLSRAASIST(XLSRAASISTConfig(**config.get("model", {}).get("xlsr", {})))
    raise ValueError(f"unsupported model: {name}")
