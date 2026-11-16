"""Legacy compatibility module; the deep-learning backbone is deferred."""

from models.base import DeepLearningDetector, DetectorOutput

__all__ = ["DeepLearningDetector", "DetectorOutput"]
