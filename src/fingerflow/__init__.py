"""Top-level helpers for Fingerflow backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Type

from .extractor import BaseExtractor, Extractor
from .matcher import BaseMatcher, Matcher


@dataclass(frozen=True)
class Backend:
    """Container describing the concrete backend implementations."""

    extractor_cls: Type[BaseExtractor]
    matcher_cls: Type[BaseMatcher]


_BACKENDS: Dict[str, Backend] = {
    "tensorflow": Backend(Extractor, Matcher),
}

try:  # pragma: no cover - optional dependency
    from fingerflow_torch import TorchExtractor, TorchMatcher
except ImportError:
    TorchExtractor = TorchMatcher = None  # type: ignore[assignment]
else:
    _BACKENDS["pytorch"] = Backend(TorchExtractor, TorchMatcher)


def get_backend(name: Optional[str] = None) -> Backend:
    """Return the registered backend for ``name``.

    Parameters
    ----------
    name:
        Identifier of the backend to use. When omitted, the value of the
        ``FINGERFLOW_BACKEND`` environment variable is used, defaulting to
        ``"tensorflow"``.

    Raises
    ------
    ValueError
        If the backend name is not registered.
    """

    if name is None:
        name = os.getenv("FINGERFLOW_BACKEND", "tensorflow")

    try:
        return _BACKENDS[name.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"Unknown Fingerflow backend: {name!r}. Available backends: {available}"
        ) from exc


__all__ = ["Backend", "get_backend"]
