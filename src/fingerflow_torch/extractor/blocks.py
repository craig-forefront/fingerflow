"""Building blocks shared across the PyTorch extractor modules."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Mish(nn.Module):
    """Implementation of the Mish activation."""

    def forward(self, input: Tensor) -> Tensor:  # type: ignore[override]
        return input * torch.tanh(F.softplus(input))


class ConvBNAct(nn.Module):
    """Convenience block mimicking the TensorFlow conv-bn-activation pattern."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        downsample: bool = False,
        activation: Optional[str] = "leaky",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        if downsample:
            stride = 2
        self.downsample = downsample
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0 if downsample else padding,
            bias=not batch_norm,
        )
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        if activation == "mish":
            self.activation: nn.Module = Mish()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if self.downsample:
            x = F.pad(x, (1, 0, 1, 0))
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block mirroring the Keras implementation."""

    def __init__(self, channels: int, hidden_channels: int, activation: str = "leaky") -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, hidden_channels, 1, activation=activation)
        self.conv2 = ConvBNAct(hidden_channels, channels, 3, activation=activation)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x + self.conv2(self.conv1(x))


class CSPBlock(nn.Module):
    """Cross Stage Partial block used by CSPDarknet53."""

    def __init__(
        self,
        channels: int,
        residual_out: int,
        repeat: int,
        *,
        residual_bottleneck: bool = False,
    ) -> None:
        super().__init__()
        hidden_channels = residual_out // 2 if residual_bottleneck else residual_out
        self.route_conv = ConvBNAct(channels, residual_out, 1, activation="mish")
        self.main_conv = ConvBNAct(channels, residual_out, 1, activation="mish")
        self.blocks = nn.ModuleList(
            [ResidualBlock(residual_out, hidden_channels, activation="mish") for _ in range(repeat)]
        )
        self.post_conv = ConvBNAct(residual_out, residual_out, 1, activation="mish")

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        route = self.route_conv(x)
        x = self.main_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_conv(x)
        return torch.cat([x, route], dim=1)
