from typing import Optional

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from IPython.display import display, clear_output

from model import SSD
from config import TrainConfig
from loss import MultiBoxLoss
from training.checkpointer import Checkpointer


class Trainer:
    
    def __init__(
        self,
        config: TrainConfig,
        model: SSD,
        criterion: MultiBoxLoss,
        optimizer: torch.optim.Optimizer,
        checkpointer: Checkpointer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpointer = checkpointer
        self.model = model.to(config.device)
        self.device = config.device
        
        self.global_step = 0        
        self.history = []

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        batch_iterator = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch:02d}",
            total=len(train_loader)
        )

        for batch in batch_iterator:
            images = batch[0].to(self.device)
            targets = [target.to(self.device) for target in batch[1]]

            predictions = self.model(images)
            
            loc_loss, conf_loss = self.criterion(predictions, targets)
            batch_loss = loc_loss + conf_loss
            
            batch_loss.backward()
            total_loss += batch_loss.item()
            
            self.global_step += 1

            if self.global_step % self.config.grad_accumulation_steps == 0:
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.scheduler:
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.optimizer.param_groups[0]["lr"]

            batch_iterator.set_postfix({
                "loc_loss": f"{loc_loss.item():.4f}",
                "conf_loss": f"{conf_loss.item():.4f}",
                "lr": f"{lr:.6e}"
            })

        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss


    @torch.no_grad()
    def eval_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        batch_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch:02d}", total=len(val_loader))
        for batch in batch_iterator:
            images = batch[0].to(self.device)
            targets = [target.to(self.device) for target in batch[1]]
            
            predictions = self.model(images)
            
            loc_loss, conf_loss = self.criterion(predictions, targets)
            batch_loss = loc_loss + conf_loss

            total_loss += batch_loss.item()
            
            batch_iterator.set_postfix({
                "loc_loss": f"{loc_loss.item():.4f}",
                "conf_loss": f"{conf_loss.item():.4f}",
            })

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 1,
        save_every: int = 3
    ) -> pd.DataFrame:
        for epoch in range(start_epoch, start_epoch + self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.eval_epoch(val_loader, epoch)

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            
            if epoch % save_every == 0:
                self.checkpointer.save(
                    epoch=epoch,
                    metrics=metrics,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler
                )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            self.history.append(row)

            clear_output(wait=True)
            display(
                pd.DataFrame(self.history).style.format({
                    "train_loss": "{:.4f}",
                    "val_loss": "{:.4f}"
                })
            )
        
        return pd.DataFrame(self.history)