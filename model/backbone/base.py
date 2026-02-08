from abc import ABC, abstractmethod
from typing import List, Any

import torch
import torch.nn as nn


class Backbone(nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def out_channels(self) -> List[int]:
        pass
