"""PyTorch reimplementation of the VerifyNet matcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from fingerflow.matcher.VerifyNet import utils as tf_utils


class EmbeddingNetwork(nn.Module):
    """Feature extractor mirroring the TensorFlow VerifyNet tower."""

    def __init__(self, input_shape: Tuple[int, int, int]) -> None:
        super().__init__()
        channels = input_shape[2]
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, input_shape[0], input_shape[1])
            flattened = self.features(dummy).view(1, -1).shape[1]
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(flattened, 16),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.flatten(x)
        return self.dense(x)


class SiameseMatcher(nn.Module):
    """PyTorch VerifyNet Siamese network."""

    def __init__(self, input_shape: Tuple[int, int, int]) -> None:
        super().__init__()
        self.embedding = EmbeddingNetwork(input_shape)
        self.bn = nn.BatchNorm1d(1)
        self.fc = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, anchor: Tensor, sample: Tensor) -> Tensor:  # type: ignore[override]
        anchor_embed = self.embedding(anchor)
        sample_embed = self.embedding(sample)
        distance = torch.norm(anchor_embed - sample_embed, dim=1, keepdim=True)
        x = self.bn(distance)
        x = self.fc(x)
        return self.sigmoid(x)


@dataclass
class TorchVerifyNet:
    """Facade around the PyTorch VerifyNet model."""

    precision: str
    weights_path: str
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        self.input_shape = tf_utils.get_input_shape(self.precision)
        self.device = torch.device(self.device)
        self.model = SiameseMatcher(self.input_shape).to(self.device)
        if self.weights_path:
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

    def __call__(self, anchor: np.ndarray, sample: np.ndarray) -> float:
        anchor_tensor = self._prepare_tensor(anchor)
        sample_tensor = self._prepare_tensor(sample)
        with torch.no_grad():
            score = self.model(anchor_tensor, sample_tensor)
        return float(score.squeeze().cpu().numpy())

    def batch(self, pairs: Iterable[Sequence[np.ndarray]]) -> np.ndarray:
        anchors, samples = zip(*pairs)
        anchor_tensor = self._prepare_tensor(np.stack(anchors))
        sample_tensor = self._prepare_tensor(np.stack(samples))
        with torch.no_grad():
            score = self.model(anchor_tensor, sample_tensor)
        return score.squeeze(-1).cpu().numpy()

    def visualize(self, file_path: str) -> None:
        dummy = torch.zeros((1, self.input_shape[2], self.input_shape[0], self.input_shape[1]), device=self.device)
        try:
            torch.onnx.export(
                self.model,
                (dummy, dummy),
                file_path,
                input_names=["anchor", "sample"],
                output_names=["score"],
                dynamic_axes={"anchor": {0: "batch"}, "sample": {0: "batch"}, "score": {0: "batch"}},
                opset_version=13,
            )
        except Exception as exc:  # pragma: no cover - ONNX export is optional during tests
            raise RuntimeError("Failed to export VerifyNet to ONNX") from exc

    def _prepare_tensor(self, array: np.ndarray) -> Tensor:
        tensor = torch.from_numpy(array).float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.permute(0, 3, 1, 2).to(self.device)
        return tensor
