"""PyTorch backend implementation for Fingerflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from fingerflow.extractor import BaseExtractor
from fingerflow.matcher import BaseMatcher

from .extractor.minutiae_net import TorchMinutiaeNet
from .matcher.verify_net import TorchVerifyNet


@dataclass
class TorchExtractor(BaseExtractor):
    """PyTorch implementation of the minutiae extractor."""

    coarse_net_path: str
    fine_net_path: str
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        self._model = TorchMinutiaeNet(
            coarse_net_path=self.coarse_net_path,
            fine_net_path=self.fine_net_path,
            device=self.device,
        )

    def extract_minutiae(self, image_data: Any) -> Dict[str, Any]:
        return self._model.extract(image_data)


@dataclass
class TorchMatcher(BaseMatcher):
    """PyTorch implementation of the fingerprint matcher."""

    weights_path: str
    precision: str = "float32"
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        self._model = TorchVerifyNet(
            precision=self.precision,
            weights_path=self.weights_path,
            device=self.device,
        )

    def verify(self, anchor: Any, sample: Any) -> Any:
        return self._model(anchor, sample)

    def verify_batch(self, pairs: Any) -> Any:
        return self._model.batch(pairs)

    def plot_model(self, file_path: str) -> None:
        self._model.visualize(file_path)


__all__ = ["TorchExtractor", "TorchMatcher", "TorchMinutiaeNet", "TorchVerifyNet"]
