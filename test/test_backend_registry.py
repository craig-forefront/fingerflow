"""Tests for the backend registration helpers."""

from __future__ import annotations

import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

import fingerflow
from fingerflow.extractor import BaseExtractor
from fingerflow.matcher import BaseMatcher


class _DummyExtractor(BaseExtractor):
    def extract_minutiae(self, image_data):  # pragma: no cover - simple stub
        return {"image": image_data}


class _DummyMatcher(BaseMatcher):
    def verify(self, anchor, sample):  # pragma: no cover - simple stub
        return anchor, sample

    def verify_batch(self, pairs):  # pragma: no cover - simple stub
        return list(pairs)

    def plot_model(self, file_path):  # pragma: no cover - simple stub
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("dummy")


class BackendRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(self._cleanup_dummy_backend)

    @staticmethod
    def _cleanup_dummy_backend() -> None:
        try:
            fingerflow.unregister_backend("dummy")
        except ValueError:
            pass

    def test_register_and_get_backend(self) -> None:
        fingerflow.register_backend("dummy", _DummyExtractor, _DummyMatcher)
        backend = fingerflow.get_backend("dummy")
        self.assertIs(backend.extractor_cls, _DummyExtractor)
        self.assertIs(backend.matcher_cls, _DummyMatcher)
        self.assertIn("dummy", fingerflow.available_backends())

    def test_register_backend_requires_subclasses(self) -> None:
        class NotExtractor:  # pragma: no cover - intentionally invalid
            pass

        with self.assertRaises(TypeError):
            fingerflow.register_backend("invalid", NotExtractor, _DummyMatcher)  # type: ignore[arg-type]

    def test_duplicate_registration_requires_overwrite(self) -> None:
        fingerflow.register_backend("dummy", _DummyExtractor, _DummyMatcher)
        with self.assertRaises(ValueError):
            fingerflow.register_backend("dummy", _DummyExtractor, _DummyMatcher)
        fingerflow.register_backend(
            "dummy", _DummyExtractor, _DummyMatcher, overwrite=True
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
