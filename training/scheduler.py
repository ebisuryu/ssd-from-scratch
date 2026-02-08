from typing import List
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


class ExponentialDecayScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1,
    ) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        epoch = max(self.last_epoch, 0)
        return [
            base_lr * (self.gamma ** epoch)
            for base_lr in self.base_lrs
        ]