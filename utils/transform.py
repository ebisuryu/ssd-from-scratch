from typing import Tuple

import torch


def center_size_to_corner_form(boxes: torch.Tensor) -> torch.Tensor:
    xy_min = boxes[:, :2] - boxes[:, 2:] / 2
    xy_max = boxes[:, :2] + boxes[:, 2:] / 2
    return torch.cat([xy_min, xy_max], dim=1)


def corner_form_to_center_size(boxes: torch.Tensor) -> torch.Tensor:
    c_xy = (boxes[:, :2] + boxes[:, 2:]) / 2
    wh = boxes[:, 2:] - boxes[:, :2]
    return torch.cat([c_xy, wh], dim=1)


def encode_boxes_to_offsets(
    matched_gt: torch.Tensor, 
    default_boxes: torch.Tensor, 
    box_variances: torch.Tensor | Tuple[float, float]
) -> torch.Tensor:
    gt = corner_form_to_center_size(matched_gt)
    g_cxcy = (gt[:, :2] - default_boxes[:, :2]) / (
        box_variances[0] * default_boxes[:, 2:]
    )
    g_wh = torch.log(gt[:, 2:] / default_boxes[:, 2:]) / box_variances[1]
    return torch.cat([g_cxcy, g_wh], dim=1)


def decode_offsets_to_boxes(
    offsets: torch.Tensor,
    default_boxes: torch.Tensor,
    box_variances: torch.Tensor | Tuple[float, float]
) -> torch.Tensor:
    cxcy = default_boxes[:, :2] + offsets[:, :2] * box_variances[0] * default_boxes[:, 2:]
    wh = default_boxes[:, 2:] * torch.exp(offsets[:, 2:] * box_variances[1])
    boxes = torch.cat([cxcy, wh], dim=1)
    return center_size_to_corner_form(boxes)