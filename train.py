import argparse

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model.ssd import SSD
from loss import MultiBoxLoss
from utils.logging import get_logger
from config import load_config_from_yaml
from training.trainer import Trainer
from training.checkpointer import Checkpointer
from training.scheduler import ExponentialDecayScheduler
from training.dataloader import get_voc_dataloaders


logger = get_logger(name="train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer from scratch")

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (last.pt or best.pt)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from training config",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Loaded training & model config")
    model_config = load_config_from_yaml(
        config_type='model',
        file_path='./configs/model/ssd300.yaml'
    )
    train_config = load_config_from_yaml(
        config_type='train',
        file_path='./configs/train.yaml'
    )
    
    if args.device is not None:
        logger.info(f"Override device → {args.device}")
        train_config.device = args.device
    
    if args.batch_size is not None:
        logger.info(f"Override batch size → {args.batch_size}")
        train_config.batch_size = args.batch_size

    if args.epochs is not None:
        logger.info(f"Override epochs → {args.epochs}")
        train_config.num_epochs = args.epochs
    
    logger.info(f"Using device: {train_config.device}")
    
    logger.info("Building dataloaders...")
    dataloaders = get_voc_dataloaders(
        batch_size=train_config.batch_size,
        local_dir="./data",
        num_workers=train_config.num_workers,
        splits=["train", "validation", "test"]
    )
    
    logger.info("Building SSD model...")
    model = SSD(config=model_config).to(train_config.device)
    
    criterion = MultiBoxLoss(
        num_classes=model_config.num_classes,
        iou_threshold=0.45,
        hard_negative_ratio=3,
        default_boxes=model.default_boxes,
        box_variances=model_config.box_variances,
        background_label=0,
        device=train_config.device
    )
    
    if train_config.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate,
            betas=train_config.betas,
            eps=train_config.eps,
            weight_decay=train_config.weight_decay,
        )
    elif train_config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            betas=train_config.betas,
            eps=train_config.eps,
            weight_decay=train_config.weight_decay,
        )
    
    if train_config.gamma:
        scheduler = ExponentialDecayScheduler(
            optimizer=optimizer,
            gamma=0.9
        )
    else:
        scheduler = None
    
    checkpointer = Checkpointer(
        checkpoint_dir=train_config.checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    
    start_epoch = 1
    
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")

        model, last_epoch, last_metrics = checkpointer.load(
            name="last" if "last" in args.resume else "best",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=train_config.device,
        )

        start_epoch = last_epoch + 1
        logger.info(
            f"Resumed training from epoch {last_epoch}, "
            f"val_loss={last_metrics.get('val_loss')}"
        )

    trainer = Trainer(
        config=train_config,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpointer=checkpointer
    )
    
    logger.info("Start training...")
    trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["validation"],
        start_epoch=start_epoch,
    )

    logger.success("Training finished successfully!")
    

if __name__ == "__main__":
    main()