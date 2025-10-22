"""Matcher interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence


class BaseMatcher(ABC):
    """Abstract interface for fingerprint matching backends."""

    @abstractmethod
    def verify(self, anchor: Any, sample: Any) -> Any:
        """Verify a pair of fingerprint samples."""
        raise NotImplementedError

    @abstractmethod
    def verify_batch(self, pairs: Iterable[Sequence[Any]]) -> Any:
        """Verify a batch of fingerprint pairs."""
        raise NotImplementedError

    @abstractmethod
    def plot_model(self, file_path: str) -> None:
        """Persist a visualization of the matcher model."""
        raise NotImplementedError


__all__ = ["BaseMatcher", "Matcher"]


def __getattr__(name: str):
    if name == "Matcher":
        from .matcher import Matcher as _Matcher

        return _Matcher
    raise AttributeError(name)
