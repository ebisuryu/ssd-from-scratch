from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.matcher import match_strategy


def log_sum_exp(x: torch.Tensor) -> torch.Tensor:
    x_max = x.data.max() 
    exp_sum = torch.sum(torch.exp(x-x_max), dim=1, keepdim=True) # [batch_size * num_boxes, 1]
    return torch.log(exp_sum) + x_max


class MultiBoxLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        iou_threshold: float,
        default_boxes: torch.Tensor,
        hard_negative_ratio: int,
        box_variances: List[float] | None = None,
        background_label: int = 0,
        device: str = 'cpu'
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.hard_negative_ratio = hard_negative_ratio
        self.background_label = background_label
        self.default_boxes = default_boxes
        self.box_variances = box_variances or []
        self.device = device

    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        targets: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_offsets, pred_scores = predictions

        batch_size = pred_offsets.size(0)
        num_boxes = self.default_boxes.size(0)

        matched_offsets = torch.zeros(
            batch_size, num_boxes, 4,
            device=self.device,
            dtype=pred_offsets.dtype,
        )
        matched_labels = torch.zeros(
            batch_size, num_boxes,
            device=self.device,
            dtype=torch.long,
        )

        for batch_idx in range(batch_size):
            with torch.no_grad():
                gt_boxes = targets[batch_idx][:, :-1]
                gt_labels = targets[batch_idx][:, -1]

                offsets, labels = match_strategy(
                    iou_threshold=self.iou_threshold,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    default_boxes=self.default_boxes,
                    box_variances=self.box_variances,
                    background_label=self.background_label,
                )

            matched_offsets[batch_idx] = offsets
            matched_labels[batch_idx] = labels

        positive_mask = matched_labels > 0
        num_positive = positive_mask.sum(dim=1, keepdim=True)

        positive_offsets_mask = positive_mask.unsqueeze(2).expand_as(pred_offsets)

        loc_loss = F.smooth_l1_loss(
            pred_offsets[positive_offsets_mask].view(-1, 4),
            matched_offsets[positive_offsets_mask].view(-1, 4),
            reduction="sum",
        )

        flat_scores = pred_scores.view(-1, self.num_classes)
        flat_labels = matched_labels.view(-1, 1)

        confidence_loss = (
            -flat_scores.gather(1, flat_labels) + log_sum_exp(flat_scores)
        ).view(batch_size, num_boxes)

        confidence_loss[positive_mask] = 0

        _, sorted_indices = confidence_loss.sort(dim=1, descending=True)
        _, rank_indices = sorted_indices.sort(dim=1)

        num_negative = torch.clamp(
            self.hard_negative_ratio * num_positive,
            max=num_boxes - 1,
        )

        negative_mask = rank_indices < num_negative.expand_as(rank_indices)

        selected_mask = positive_mask | negative_mask
        selected_scores = pred_scores[selected_mask].view(-1, self.num_classes)
        selected_labels = matched_labels[selected_mask]

        conf_loss = F.cross_entropy(
            selected_scores,
            selected_labels,
            reduction="sum",
        )

        normalizer = num_positive.sum().clamp(min=1)

        return loc_loss / normalizer, conf_loss / normalizer