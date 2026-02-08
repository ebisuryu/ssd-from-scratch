import torch
import numpy as np


def intersect(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    A = boxes_a.size(0)
    B = boxes_b.size(0)

    max_xy = torch.min(
        boxes_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        boxes_b[:, 2:].unsqueeze(0).expand(A, B, 2)
    )
    min_xy = torch.max(
        boxes_a[:, :2].unsqueeze(1).expand(A, B, 2),
        boxes_b[:, :2].unsqueeze(0).expand(A, B, 2)
    )

    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    inter = intersect(boxes_a, boxes_b)

    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) *
              (boxes_a[:, 3] - boxes_a[:, 1])).unsqueeze(1)

    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) *
              (boxes_b[:, 3] - boxes_b[:, 1])).unsqueeze(0)

    union = area_a + area_b - inter
    return inter / union