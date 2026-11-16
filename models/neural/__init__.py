"""Neural detector implementations."""

from .baseline_cnn import BaselineCNN, BaselineCNNConfig
from .xlsr_aasist import XLSRAASIST, XLSRAASISTConfig

__all__ = ["BaselineCNN", "BaselineCNNConfig", "XLSRAASIST", "XLSRAASISTConfig"]
