import argparse
import torch
from tqdm import tqdm

from inference import ssd_inference
from utils.logging import get_logger
from config import load_config_from_yaml
from training.dataloader import get_voc_dataloaders
from model.ssd import SSD
from training.evaluator import DetectionEvaluator


logger = get_logger(name="evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Detection model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model weights",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="",
    )
    
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="",
    )
    
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    logger.info("Loaded training & model config")
    model_config = load_config_from_yaml(
        config_type="model",
        file_path="./configs/model/ssd300.yaml"
    )
    evaluate_config = load_config_from_yaml(
        config_type="evaluate",
        file_path="./configs/evaluate.yaml"
    )
    
    if args.checkpoint is not None:
        logger.info(f"Override checkpoint → {args.checkpoint}")
        evaluate_config.checkpoint = args.checkpoint
    
    if args.batch_size is not None:
        logger.info(f"Override batch_size → {args.batch_size}")
        evaluate_config.batch_size = args.batch_size
    
    if args.num_workers is not None:
        logger.info(f"Override num_workers → {args.num_workers}")
        evaluate_config.num_workers = args.num_workers
    
    if args.conf_threshold is not None:
        logger.info(f"Override conf_threshold → {args.conf_threshold}")
        evaluate_config.conf_threshold = args.conf_threshold
    
    if args.iou_threshold is not None:
        logger.info(f"Override iou_threshold → {args.iou_threshold}")
        evaluate_config.iou_threshold = args.iou_threshold
    
    if args.device is not None:
        logger.info(f"Override device → {args.device}")
        evaluate_config.device = args.device
    
    model = SSD(config=model_config).to(evaluate_config.device)
    
    logger.info(f"Loading checkpoint from {evaluate_config.checkpoint}")
    checkpoint = torch.load(
        evaluate_config.checkpoint,
        map_location=evaluate_config.device,
        weights_only=False
    )
    model.load_state_dict(
        checkpoint["model_state"],
        strict=False,
    )
    model.eval()

    dataloaders = get_voc_dataloaders(
        batch_size=evaluate_config.batch_size,
        local_dir="./data",
        num_workers=evaluate_config.num_workers,
        splits=["test"]
    )

    evaluator = DetectionEvaluator(
        box_format="xyxy",
        conf_threshold=evaluate_config.conf_threshold
    )
    
    test_loader = dataloaders["test"]

    batch_iterator = tqdm(
        test_loader,
        desc="Evaluating",
        total=len(test_loader),
    )

    logger.info("Dectecting with model...")
    image_counter = 0
    for batch in batch_iterator:
        images = batch[0].to(evaluate_config.device)
        targets = [t.to(evaluate_config.device) for t in batch[1]]

        with torch.no_grad():
            loc, conf = model(images)

            outputs = ssd_inference(
                loc=loc, 
                conf=conf,
                default_boxes=model.default_boxes,
                num_classes=model_config.num_classes,
                background_label=0,
                top_k=200,
                conf_threshold=evaluate_config.conf_threshold,
                iou_threshold=evaluate_config.iou_threshold,
                box_variances=model_config.box_variances,
            )

        batch_size = images.size(0)

        evaluator.update(
            predictions=outputs,
            targets=targets
        )

        image_counter += batch_size
    
    logger.info("Computing metrics...")
    scores = evaluator.compute()
    
    logger.info("Evaluation scores")
    for k, v in scores.items():
        logger.info(f"{k:25s}: {v:.4f}")


if __name__ == "__main__":
    main()