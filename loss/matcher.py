from typing import Tuple

import torch

from utils import (
    encode_boxes_to_offsets, jaccard, center_size_to_corner_form
)


def match_strategy(
    iou_threshold: float,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    default_boxes: torch.Tensor,
    box_variances: torch.Tensor | Tuple[float, float],
    background_label: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    corner_defaults = center_size_to_corner_form(default_boxes)
    overlaps = jaccard(gt_boxes, corner_defaults)  # [M, N]

    best_dbox_overlap, best_dbox_idx = overlaps.max(dim=1)
    best_gt_overlap, best_gt_idx = overlaps.max(dim=0)
    best_gt_idx.scatter_(
        0,
        best_dbox_idx,
        torch.arange(best_dbox_idx.size(0), device=best_gt_idx.device),
    )
    best_gt_overlap.index_fill_(0, best_dbox_idx, 2.0)

    matched_boxes = gt_boxes[best_gt_idx]
    conf = gt_labels[best_gt_idx] + 1
    conf[best_gt_overlap < iou_threshold] = background_label

    loc = encode_boxes_to_offsets(
        matched_gt=matched_boxes, 
        default_boxes=default_boxes, 
        box_variances=box_variances
    )
    
    return loc, conf