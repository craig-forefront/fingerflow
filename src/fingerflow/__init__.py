"""Top-level helpers for Fingerflow backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

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


def get_backend(name: str) -> Backend:
    """Return the registered backend for ``name``.

    Parameters
    ----------
    name:
        Identifier of the backend to use. Currently ``"tensorflow"`` is
        available. The lookup is case-insensitive.

    Raises
    ------
    ValueError
        If the backend name is not registered.
    """

    try:
        return _BACKENDS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown Fingerflow backend: {name!r}") from exc


__all__ = ["Backend", "get_backend"]
