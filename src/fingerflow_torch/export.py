"""ONNX export helpers for the PyTorch backend."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import onnxruntime as ort
import torch

from .extractor.minutiae_net import TorchCoarseNet
from .matcher.verify_net import SiameseMatcher


def export_extractor(model: TorchCoarseNet, sample: torch.Tensor, output: Path) -> None:
    model.eval()
    torch.onnx.export(
        model,
        sample,
        output,
        input_names=["image"],
        output_names=["boxes", "object", "classes", "pred_wh"],
        opset_version=13,
        dynamic_axes={"image": {0: "batch"}, "boxes": {0: "batch"}, "object": {0: "batch"}},
    )


def export_matcher(model: SiameseMatcher, sample: torch.Tensor, output: Path) -> None:
    model.eval()
    torch.onnx.export(
        model,
        (sample, sample),
        output,
        input_names=["anchor", "sample"],
        output_names=["score"],
        opset_version=13,
        dynamic_axes={"anchor": {0: "batch"}, "sample": {0: "batch"}, "score": {0: "batch"}},
    )


def verify_onnx(model_path: Path, inputs: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    session = ort.InferenceSession(str(model_path))
    input_names = [inp.name for inp in session.get_inputs()]
    feed_dict = {name: array for name, array in zip(input_names, inputs)}
    return session.run(None, feed_dict)
