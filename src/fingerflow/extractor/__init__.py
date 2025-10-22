"""Extractor interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseExtractor(ABC):
    """Abstract interface for minutiae extraction backends."""

    @abstractmethod
    def extract_minutiae(self, image_data: Any) -> Dict[str, Any]:
        """Extract minutiae information from a fingerprint image."""
        raise NotImplementedError

__all__ = ["BaseExtractor", "Extractor"]


def __getattr__(name: str):
    if name == "Extractor":
        from .extractor import Extractor as _Extractor

        return _Extractor
    raise AttributeError(name)
