"""Smoke tests for the PyTorch backend export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fingerflow_torch.export import export_extractor, export_matcher, verify_onnx
from fingerflow_torch.extractor.minutiae_net import TorchCoarseNet
from fingerflow_torch.matcher.verify_net import SiameseMatcher


def _onnx_path(tmp_path: Path, name: str) -> Path:
    return tmp_path / f"{name}.onnx"


def test_matcher_export_roundtrip(tmp_path: Path) -> None:
    model = SiameseMatcher((64, 64, 1))
    sample = torch.randn(2, 1, 64, 64)
    path = _onnx_path(tmp_path, "matcher")
    export_matcher(model, sample, path)

    torch_output = model(sample, sample).detach().cpu().numpy()
    onnx_outputs = verify_onnx(path, [sample.cpu().numpy(), sample.cpu().numpy()])
    assert np.allclose(torch_output, onnx_outputs[0], atol=1e-5)


def test_extractor_export_roundtrip(tmp_path: Path) -> None:
    model = TorchCoarseNet()
    sample = torch.randn(1, 3, 416, 416)
    path = _onnx_path(tmp_path, "extractor")
    export_extractor(model, sample, path)

    torch_outputs = model(sample)
    onnx_outputs = verify_onnx(path, [sample.cpu().numpy()])
    assert len(onnx_outputs) == len(torch_outputs)
    for torch_tensor, onnx_array in zip(torch_outputs, onnx_outputs):
        assert np.allclose(torch_tensor.detach().cpu().numpy(), onnx_array, atol=1e-5)
