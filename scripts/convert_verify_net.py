"""Convert VerifyNet TensorFlow weights to the PyTorch backend."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
import torch

from fingerflow.matcher.VerifyNet import verify_net_model, utils as tf_utils
from fingerflow_torch.matcher.verify_net import SiameseMatcher


def _transpose_conv(kernel: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(kernel, (3, 2, 0, 1)))


def _transpose_dense(kernel: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(kernel.T)


def convert_verify_net(weights: Path, output: Path, precision: str) -> None:
    tf_model = verify_net_model.get_verify_net_model(precision, str(weights))
    input_shape = tf_utils.get_input_shape(precision)
    torch_model = SiameseMatcher(input_shape)
    state_dict = torch_model.state_dict()

    layer_map: Dict[str, str] = {
        "embedding.features.0": "siamese_matcher/embedding_network/conv2d",
        "embedding.features.3": "siamese_matcher/embedding_network/conv2d_1",
        "embedding.dense.0": "siamese_matcher/embedding_network/dense",
        "bn": "batch_normalization",
        "fc": "dense_1",
    }

    for torch_name, tf_layer_name in layer_map.items():
        layer = tf_model.get_layer(tf_layer_name)
        weights_list = layer.get_weights()
        if not weights_list:
            continue
        if isinstance(layer, tf.keras.layers.Conv2D):
            weight, bias = weights_list
            state_dict[f"{torch_name}.weight"] = _transpose_conv(weight)
            state_dict[f"{torch_name}.bias"] = torch.from_numpy(bias)
        elif isinstance(layer, tf.keras.layers.Dense):
            weight, bias = weights_list
            state_dict[f"{torch_name}.weight"] = _transpose_dense(weight)
            state_dict[f"{torch_name}.bias"] = torch.from_numpy(bias)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            gamma, beta, mean, variance = weights_list
            state_dict[f"{torch_name}.weight"] = torch.from_numpy(gamma)
            state_dict[f"{torch_name}.bias"] = torch.from_numpy(beta)
            state_dict[f"{torch_name}.running_mean"] = torch.from_numpy(mean)
            state_dict[f"{torch_name}.running_var"] = torch.from_numpy(variance)
        else:
            raise TypeError(f"Unsupported layer type: {type(layer)!r}")

    torch_model.load_state_dict(state_dict)
    torch.save(torch_model.state_dict(), output)
    print(f"PyTorch weights saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Path to the TensorFlow VerifyNet checkpoint")
    parser.add_argument("output", type=Path, help="Destination for the converted PyTorch checkpoint")
    parser.add_argument("--precision", default="float32", help="Precision used to instantiate the TensorFlow model")
    args = parser.parse_args()
    convert_verify_net(args.weights, args.output, args.precision)


if __name__ == "__main__":
    main()
