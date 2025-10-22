"""Top-level helpers for Fingerflow backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

from .extractor import BaseExtractor
from .matcher import BaseMatcher


@dataclass(frozen=True)
class Backend:
    """Container describing the concrete backend implementations."""

    extractor_cls: Type[BaseExtractor]
    matcher_cls: Type[BaseMatcher]


_BACKENDS: Dict[str, Backend] = {}

try:  # pragma: no cover - optional heavy dependency chain
    from .extractor import Extractor
    from .matcher import Matcher
except ModuleNotFoundError:
    Extractor = Matcher = None  # type: ignore[assignment]
else:
    _BACKENDS["tensorflow"] = Backend(Extractor, Matcher)

try:  # pragma: no cover - optional dependency
    from fingerflow_torch import TorchExtractor, TorchMatcher
except ImportError:
    TorchExtractor = TorchMatcher = None  # type: ignore[assignment]
else:
    _BACKENDS["pytorch"] = Backend(TorchExtractor, TorchMatcher)


def register_backend(
    name: str,
    extractor_cls: Type[BaseExtractor],
    matcher_cls: Type[BaseMatcher],
    *,
    overwrite: bool = False,
) -> None:
    """Register a backend implementation.

    Parameters
    ----------
    name:
        Human-friendly identifier of the backend.
    extractor_cls:
        Concrete :class:`~fingerflow.extractor.BaseExtractor` subclass
        implementing minutiae extraction.
    matcher_cls:
        Concrete :class:`~fingerflow.matcher.BaseMatcher` subclass
        implementing fingerprint matching.
    overwrite:
        When ``True`` an existing backend with the same name will be
        replaced. The default ``False`` raises :class:`ValueError` if the
        backend name already exists.

    Raises
    ------
    TypeError
        If ``extractor_cls`` or ``matcher_cls`` are not subclasses of the
        expected abstract base classes.
    ValueError
        When attempting to overwrite an existing backend without explicitly
        opting in via ``overwrite``.
    """

    if not issubclass(extractor_cls, BaseExtractor):
        raise TypeError("extractor_cls must inherit from BaseExtractor")
    if not issubclass(matcher_cls, BaseMatcher):
        raise TypeError("matcher_cls must inherit from BaseMatcher")

    key = name.lower()
    if not overwrite and key in _BACKENDS:
        raise ValueError(f"Backend {name!r} is already registered")

    _BACKENDS[key] = Backend(extractor_cls, matcher_cls)


def unregister_backend(name: str) -> None:
    """Remove a previously registered backend.

    Parameters
    ----------
    name:
        Identifier previously passed to :func:`register_backend`.

    Raises
    ------
    ValueError
        If the backend name is unknown.
    """

    key = name.lower()
    try:
        del _BACKENDS[key]
    except KeyError as exc:  # pragma: no cover - defensive branch
        available = ", ".join(sorted(_BACKENDS)) or "<none>"
        raise ValueError(
            f"Cannot unregister unknown backend {name!r}. Available backends: {available}"
        ) from exc


def available_backends() -> Tuple[str, ...]:
    """Return a tuple of registered backend names sorted alphabetically."""

    return tuple(sorted(_BACKENDS))


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


__all__ = [
    "Backend",
    "available_backends",
    "get_backend",
    "register_backend",
    "unregister_backend",
]
