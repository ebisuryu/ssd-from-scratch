from typing import List
import argparse

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from model import SSD
from config import load_config_from_yaml
from utils import nms, visualize_bboxes, decode_offsets_to_boxes


def ssd_inference(
    loc: torch.Tensor,
    conf: torch.Tensor,
    default_boxes: torch.Tensor,
    num_classes: int,
    background_label: int,
    top_k: int,
    conf_threshold: float,
    iou_threshold: float,
    box_variances: List[float],
) -> torch.Tensor:
    
    loc, conf, default_boxes = loc.detach(), conf.detach(), default_boxes.detach() 
    conf = F.softmax(conf, dim=-1)
    
    batch_size = loc.size(0)
    conf_preds = conf.transpose(2, 1)
    output = torch.zeros(batch_size, num_classes, top_k, 5, device=loc.device)

    for b in range(batch_size):
        decoded_boxes = decode_offsets_to_boxes(loc[b], default_boxes, box_variances)
        scores_per_class = conf_preds[b]

        for cls_id in range(num_classes):
            if cls_id == background_label:
                continue

            scores = scores_per_class[cls_id]
            mask = scores > conf_threshold

            if mask.sum() == 0:
                continue

            cls_scores = scores[mask]
            cls_boxes = decoded_boxes[mask]

            indices, count = nms(
                boxes=cls_boxes,
                scores=cls_scores,
                overlap=iou_threshold,
                top_k=top_k,
            )

            keep = indices[:count]
            output[b, cls_id, :count] = torch.cat(
                [cls_scores[keep].unsqueeze(1), cls_boxes[keep]],
                dim=1,
            )

    return output


def preprocess_image(
    image_path: str,
    image_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(image_path)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(
        image,
        (image_size, image_size),
    ).astype(np.float32)

    x -= (104.0, 117.0, 123.0)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    return x, rgb_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSD inference",)
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./weights/checkpoint.pt",
        help="Path to model checkpoint file",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Maximum number of detections per class",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--show",
        default=True,
        help="Visualize detection results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    model_config = load_config_from_yaml(
        config_type="model",
        file_path="./configs/model/ssd300.yaml",
    )
    data_config = load_config_from_yaml(
        config_type="data",
        file_path="./configs/data/voc.yaml",
    )

    model = SSD(
        config=model_config,
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        checkpoint["model_state"],
        strict=False,
    )
    model.eval()

    x, rgb_image = preprocess_image(
        args.image_path,
        model_config.image_size,
    )
    x = x.to(device)

    with torch.no_grad():
        loc, conf = model(x)
        default_boxes = model.default_boxes

        results = ssd_inference(
            loc=loc,
            conf=conf,
            default_boxes=default_boxes,
            num_classes=model_config.num_classes,
            background_label=data_config.background_label,
            top_k=args.top_k,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            box_variances=model_config.box_variances,
        )

    if args.show:
        visualize_bboxes(
            rgb_image=rgb_image,
            bboxes=results,
            labels=data_config.labels,
            conf_threshold=args.conf_threshold,
        )


if __name__ == "__main__":
    main()