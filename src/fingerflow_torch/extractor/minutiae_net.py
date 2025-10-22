"""PyTorch MinutiaeNet implementation mirroring the TensorFlow graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import batched_nms

from fingerflow.extractor.CoreNet import constants as core_constants

from .blocks import CSPBlock, ConvBNAct


class CSPDarknet53(nn.Module):
    """Backbone network shared by the coarse extractor."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNAct(in_channels, 32, 3, activation="mish"),
            ConvBNAct(32, 64, 3, downsample=True, activation="mish"),
        )
        self.csp1 = CSPBlock(64, 64, repeat=1, residual_bottleneck=True)
        self.csp1_reduction = ConvBNAct(128, 64, 1, activation="mish")
        self.down1 = ConvBNAct(64, 128, 3, downsample=True, activation="mish")

        self.csp2 = CSPBlock(128, 64, repeat=2)
        self.csp2_reduction = ConvBNAct(128, 128, 1, activation="mish")
        self.down2 = ConvBNAct(128, 256, 3, downsample=True, activation="mish")

        self.csp3 = CSPBlock(256, 128, repeat=8)
        self.csp3_reduction = ConvBNAct(256, 256, 1, activation="mish")
        self.down3 = ConvBNAct(256, 512, 3, downsample=True, activation="mish")

        self.csp4 = CSPBlock(512, 256, repeat=8)
        self.csp4_reduction = ConvBNAct(512, 512, 1, activation="mish")
        self.down4 = ConvBNAct(512, 1024, 3, downsample=True, activation="mish")

        self.csp5 = CSPBlock(1024, 512, repeat=4)
        self.final_trunk = nn.Sequential(
            ConvBNAct(1024, 1024, 1, activation="mish"),
            ConvBNAct(1024, 512, 1),
            ConvBNAct(512, 1024, 3),
            ConvBNAct(1024, 512, 1),
        )
        self.pool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.pool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.final = nn.Sequential(
            ConvBNAct(2048, 512, 1),
            ConvBNAct(512, 1024, 3),
            ConvBNAct(1024, 512, 1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore[override]
        x = self.stage1(x)
        x = self.csp1(x)
        x = self.csp1_reduction(x)
        x = self.down1(x)

        x = self.csp2(x)
        x = self.csp2_reduction(x)
        x = self.down2(x)

        x = self.csp3(x)
        route0 = self.csp3_reduction(x)
        x = self.down3(route0)

        x = self.csp4(x)
        route1 = self.csp4_reduction(x)
        x = self.down4(route1)

        x = self.csp5(x)
        x = self.final_trunk(x)

        pooled = torch.cat([self.pool13(x), self.pool9(x), self.pool5(x), x], dim=1)
        route2 = self.final(pooled)
        return route0, route1, route2


class YoloV4Neck(nn.Module):
    """Implementation of the YOLOv4 neck used in the coarse model."""

    def __init__(self, num_classes: int = core_constants.NUM_CLASSES) -> None:
        super().__init__()
        self.backbone = CSPDarknet53()

        self.conv_for_route2 = nn.Sequential(
            ConvBNAct(512, 256, 1),
        )
        self.up_route2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_route1 = ConvBNAct(512, 256, 1)
        self.stage_route1 = nn.Sequential(
            ConvBNAct(512, 256, 1),
            ConvBNAct(256, 512, 3),
            ConvBNAct(512, 256, 1),
            ConvBNAct(256, 512, 3),
            ConvBNAct(512, 256, 1),
        )
        self.conv_for_route1 = nn.Sequential(
            ConvBNAct(256, 128, 1),
        )
        self.up_route1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_route0 = ConvBNAct(256, 128, 1)
        self.stage_route0 = nn.Sequential(
            ConvBNAct(256, 128, 1),
            ConvBNAct(128, 256, 3),
            ConvBNAct(256, 128, 1),
            ConvBNAct(128, 256, 3),
            ConvBNAct(256, 128, 1),
        )

        self.conv_sbbox = nn.Sequential(
            ConvBNAct(128, 256, 3),
            ConvBNAct(256, 3 * (num_classes + 5), 1, activation=None, batch_norm=False),
        )
        self.down_route0 = nn.Sequential(
            ConvBNAct(128, 256, 3, downsample=True),
        )
        self.stage_middle = nn.Sequential(
            ConvBNAct(512, 256, 1),
            ConvBNAct(256, 512, 3),
            ConvBNAct(512, 256, 1),
            ConvBNAct(256, 512, 3),
            ConvBNAct(512, 256, 1),
        )
        self.conv_mbbox = nn.Sequential(
            ConvBNAct(256, 512, 3),
            ConvBNAct(512, 3 * (num_classes + 5), 1, activation=None, batch_norm=False),
        )
        self.down_route1 = nn.Sequential(
            ConvBNAct(256, 512, 3, downsample=True),
        )
        self.stage_large = nn.Sequential(
            ConvBNAct(1024, 512, 1),
            ConvBNAct(512, 1024, 3),
            ConvBNAct(1024, 512, 1),
            ConvBNAct(512, 1024, 3),
            ConvBNAct(1024, 512, 1),
        )
        self.conv_lbbox = nn.Sequential(
            ConvBNAct(512, 1024, 3),
            ConvBNAct(1024, 3 * (num_classes + 5), 1, activation=None, batch_norm=False),
        )

    def forward(self, x: Tensor) -> List[Tensor]:  # type: ignore[override]
        route0, route1, route2 = self.backbone(x)

        x = self.conv_for_route2(route2)
        x = self.up_route2(x)
        x = torch.cat([self.conv_route1(route1), x], dim=1)
        x = self.stage_route1(x)

        route1_processed = x
        x = self.conv_for_route1(x)
        x = self.up_route1(x)
        x = torch.cat([self.conv_route0(route0), x], dim=1)
        x = self.stage_route0(x)

        route0_processed = x
        small = self.conv_sbbox(x)

        x = self.down_route0(route0_processed)
        x = torch.cat([x, route1_processed], dim=1)
        x = self.stage_middle(x)

        route1_processed = x
        medium = self.conv_mbbox(x)

        x = self.down_route1(route1_processed)
        x = torch.cat([x, route2], dim=1)
        x = self.stage_large(x)
        large = self.conv_lbbox(x)

        return [small, medium, large]


def _reshape_predictions(pred: Tensor, grid_size: int, num_classes: int) -> Tensor:
    bs = pred.shape[0]
    pred = pred.view(bs, 3, num_classes + 5, grid_size, grid_size)
    pred = pred.permute(0, 3, 4, 1, 2).contiguous()
    return pred


def _decode_predictions(
    pred: Tensor,
    anchors: Tensor,
    grid_size: int,
    stride: int,
    xyscale: float,
    num_classes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    pred = _reshape_predictions(pred, grid_size, num_classes)
    box_xy, box_wh, obj_prob, class_prob = torch.split(pred, (2, 2, 1, num_classes), dim=-1)
    box_xy = torch.sigmoid(box_xy)
    obj_prob = torch.sigmoid(obj_prob)
    class_prob = torch.sigmoid(class_prob)
    pred_box_xywh = torch.cat((box_xy, box_wh), dim=-1)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size, device=pred.device),
        torch.arange(grid_size, device=pred.device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2).float()
    box_xy = ((box_xy * xyscale) - 0.5 * (xyscale - 1) + grid) * stride
    box_wh = torch.exp(box_wh) * anchors
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    pred_box_x1y1x2y2 = torch.cat([box_x1y1, box_x2y2], dim=-1)
    return pred_box_x1y1x2y2, obj_prob, class_prob, pred_box_xywh


class TorchCoarseNet(nn.Module):
    """PyTorch counterpart of the TensorFlow CoarseNet."""

    def __init__(self, num_classes: int = core_constants.NUM_CLASSES) -> None:
        super().__init__()
        self.neck = YoloV4Neck(num_classes=num_classes)
        self.anchors = torch.tensor(core_constants.ANCHORS, dtype=torch.float32).view(3, 3, 2)
        self.num_classes = num_classes
        self.xy_scales = core_constants.XY_SCALE
        self.strides = [8, 16, 32]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore[override]
        outputs = self.neck(x)
        decoded: List[Tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for pred, anchors, stride, scale, grid_size in zip(
            outputs,
            self.anchors,
            self.strides,
            self.xy_scales,
            [52, 26, 13],
        ):
            decoded.append(
                _decode_predictions(
                    pred,
                    anchors.to(pred.device),
                    grid_size,
                    stride,
                    scale,
                    self.num_classes,
                )
            )
        boxes = [item[0] for item in decoded]
        obj = [item[1] for item in decoded]
        classes = [item[2] for item in decoded]
        pred_wh = [item[3] for item in decoded]
        return (
            torch.cat([b.view(b.size(0), -1, 4) for b in boxes], dim=1),
            torch.cat([o.view(o.size(0), -1, 1) for o in obj], dim=1),
            torch.cat([c.view(c.size(0), -1, self.num_classes) for c in classes], dim=1),
            torch.cat([w.view(w.size(0), -1, 4) for w in pred_wh], dim=1),
        )


def _apply_nms(
    boxes: Tensor,
    scores: Tensor,
    classes: Tensor,
    max_total_size: int = 100,
    score_threshold: float = core_constants.SCORE_THRESHOLD,
    iou_threshold: float = core_constants.IOU_THRESHOLD,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    batch = boxes.shape[0]
    num_classes = classes.shape[-1]
    results = []
    for batch_idx in range(batch):
        box = boxes[batch_idx] / core_constants.INPUT_SHAPE[0]
        score = scores[batch_idx]
        cls_prob = classes[batch_idx]
        score = score * cls_prob
        score = score.view(-1)
        class_ids = torch.arange(num_classes, device=boxes.device).repeat(box.size(0))
        valid = score > score_threshold
        if valid.any():
            filtered_boxes = box[valid]
            filtered_scores = score[valid]
            filtered_classes = class_ids[valid]
            keep = batched_nms(
                filtered_boxes,
                filtered_scores,
                filtered_classes,
                iou_threshold,
            )
            keep = keep[:max_total_size]
            filtered_boxes = filtered_boxes[keep]
            filtered_scores = filtered_scores[keep]
            filtered_classes = filtered_classes[keep]
        else:
            filtered_boxes = torch.zeros((0, 4), device=boxes.device)
            filtered_scores = torch.zeros((0,), device=boxes.device)
            filtered_classes = torch.zeros((0,), device=boxes.device)
        valid_count = torch.tensor([filtered_boxes.size(0)], device=boxes.device)
        pad_size = max_total_size - filtered_boxes.size(0)
        if pad_size > 0:
            filtered_boxes = F.pad(filtered_boxes, (0, 0, 0, pad_size))
            filtered_scores = F.pad(filtered_scores, (0, pad_size))
            filtered_classes = F.pad(filtered_classes, (0, pad_size))
        results.append((filtered_boxes, filtered_scores, filtered_classes, valid_count))
    stacked = [torch.stack(items, dim=0) for items in zip(*results)]
    return stacked[0], stacked[1], stacked[2], stacked[3]


class TorchFineNet(nn.Module):
    """Simplified FineNet counterpart based on the public MinutiaeNet implementation."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(1, 32, 3),
            nn.MaxPool2d(2),
            ConvBNAct(32, 64, 3),
            nn.MaxPool2d(2),
            ConvBNAct(64, 128, 3),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 25 * 25, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = self.features(x)
        return self.classifier(x)


@dataclass
class TorchMinutiaeNet:
    """High level wrapper replicating the TensorFlow MinutiaeNet pipeline."""

    coarse_net_path: str
    fine_net_path: str
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.coarse_model = TorchCoarseNet().to(self.device)
        self.fine_model = TorchFineNet().to(self.device)
        if self.coarse_net_path:
            state = torch.load(self.coarse_net_path, map_location=self.device)
            self.coarse_model.load_state_dict(state)
        if self.fine_net_path:
            state = torch.load(self.fine_net_path, map_location=self.device)
            self.fine_model.load_state_dict(state)
        self.coarse_model.eval()
        self.fine_model.eval()

    def _coarse_forward(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tensor = torch.from_numpy(image).float()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor.to(self.device)
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        with torch.no_grad():
            boxes, obj, cls_prob, pred_wh = self.coarse_model(tensor)
        return (
            boxes.cpu().numpy(),
            obj.cpu().numpy(),
            cls_prob.cpu().numpy(),
            pred_wh.cpu().numpy(),
        )

    def extract_minutiae_points(self, image: np.ndarray, original_image: np.ndarray | None = None) -> np.ndarray:
        # Placeholder logic: actual minutiae extraction requires the original TensorFlow utilities.
        raise NotImplementedError("Full minutiae extraction pipeline is not yet implemented in PyTorch.")

    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        boxes, obj, cls_prob, _ = self._coarse_forward(image)
        return {
            "boxes": boxes,
            "scores": obj * cls_prob,
        }
