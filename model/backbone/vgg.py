from typing import List, Union

import torch
import torch.nn as nn

from model.backbone.base import Backbone


class VGGBackbone(Backbone):

    def __init__(
        self, 
        architecture: List[Union[int, str]], 
        in_channels: int = 3, 
        batch_norm: bool = False
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        
        for spec in architecture:
            if spec == 'M':
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif spec == 'C':
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            else:
                conv2d = nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=spec, 
                    kernel_size=3, 
                    padding=1
                )
                self.layers.append(conv2d)
                if batch_norm:
                    self.layers.append(nn.BatchNorm2d(spec))
                self.layers.append(nn.ReLU(inplace=True))
                in_channels = spec

        self.layers += [
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == 22:
                outputs.append(x)

        outputs.append(x)
        return outputs

    def out_channels(self) -> List[int]:
        return [
            self.layers[idx].out_channels 
            for idx in [21, -2]
        ]