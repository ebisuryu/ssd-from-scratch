from collections import defaultdict
from typing import List, Dict

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from utils.iou import jaccard


class DetectionEvaluator:

    def __init__(
        self,
        box_format: str = "xyxy",
        conf_threshold: float = 0.01,
        iou_thresholds: None | List[float] = None,
        pr_iou_threshold: float = 0.5,   # 👈 thêm
    ) -> None:

        if iou_thresholds is None:
            iou_thresholds = torch.arange(0.5, 0.96, 0.05).tolist()

        self.iou_thresholds = iou_thresholds
        self.iou_threshold = pr_iou_threshold
        self.conf_threshold = conf_threshold

        self.map_metric = MeanAveragePrecision(
            box_format=box_format,
            iou_thresholds=iou_thresholds,
        )

        self.reset()

    def reset(self) -> None:
        self.map_metric.reset()
        self.stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    def update(self, predictions: torch.Tensor, targets: List[torch.Tensor]) -> None:
        processed_preds = []
        processed_targets = []

        batch_size, num_classes_with_bg, num_boxes, _ = predictions.shape

        for b in range(batch_size):
            img_preds = predictions[b]

            all_boxes = []
            all_scores = []
            all_labels = []

            for cls_idx in range(1, num_classes_with_bg):  # skip background (0)
                scores = img_preds[cls_idx, :, 0]
                boxes = img_preds[cls_idx, :, 1:]

                mask = scores > self.conf_threshold
                if mask.any():
                    all_boxes.append(boxes[mask])
                    all_scores.append(scores[mask])
                    all_labels.append(
                        torch.full(
                            (mask.sum(),),
                            cls_idx - 1, 
                            device=predictions.device,
                        )
                    )

            if len(all_boxes) > 0:
                processed_preds.append(
                    {
                        "boxes": torch.cat(all_boxes, dim=0),
                        "scores": torch.cat(all_scores, dim=0),
                        "labels": torch.cat(all_labels, dim=0).int(),
                    }
                )
            else:
                processed_preds.append(
                    {
                        "boxes": torch.empty((0, 4), device=predictions.device),
                        "scores": torch.empty((0,), device=predictions.device),
                        "labels": torch.empty((0,), device=predictions.device).int(),
                    }
                )

            target = targets[b]
            processed_targets.append(
                {
                    "boxes": target[:, :4],
                    "labels": target[:, 4].int(),
                }
            )
            
        self.map_metric.update(processed_preds, processed_targets)
        self._update_prf(processed_preds, processed_targets)

    def _update_prf(self, predictions: torch.Tensor, targets: List[torch.Tensor]) -> None:
        for prediction, target in zip(predictions, targets):
            classes = torch.unique(
                torch.cat([prediction["labels"], target["labels"]], dim=0)
            )

            for cls in classes:
                cls = int(cls.item())

                prediction_mask = prediction["labels"] == cls
                target_mask = target["labels"] == cls

                prediction_boxes = prediction["boxes"][prediction_mask]
                target_boxes = target["boxes"][target_mask]

                if len(target_boxes) == 0:
                    self.stats[cls]["fp"] += len(prediction_boxes)
                    continue

                if len(prediction_boxes) == 0:
                    self.stats[cls]["fn"] += len(target_boxes)
                    continue

                ious = jaccard(prediction_boxes, target_boxes)

                matched_gt = set()
                for i in range(len(prediction_boxes)):
                    max_iou, idx = ious[i].max(0)
                    if max_iou >= self.iou_threshold and idx.item() not in matched_gt:
                        self.stats[cls]["tp"] += 1
                        matched_gt.add(idx.item())
                    else:
                        self.stats[cls]["fp"] += 1

                self.stats[cls]["fn"] += len(target_boxes) - len(matched_gt)

    def compute(self) -> Dict[str, float]:
        map_result = self.map_metric.compute()
        precisions, recalls, f1s = [], [], []

        for s in self.stats.values():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return  {
            "mAP": map_result["map"].item(),
            "mAP@50": map_result["map_50"].item(),
            "mAP@75": map_result["map_75"].item(),
            "Precision@0.5": float(sum(precisions) / max(len(precisions), 1)),
            "Recall@0.5": float(sum(recalls) / max(len(recalls), 1)),
            "F1@0.5": float(sum(f1s) / max(len(f1s), 1)),
        }