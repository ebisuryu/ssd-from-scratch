from typing import List, Union

import torch
import torch.nn as nn


class ExtraLayers(nn.Module):

    def __init__(
        self,
        architecture: List[Union[int, str]],
        in_channels: int,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        use_3x3_kernel = False   
        skip_current = False  

        for idx, spec in enumerate(architecture):
            kernel_size = 3 if use_3x3_kernel else 1

            if spec == 'S' and not skip_current:
                self.layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=architecture[idx + 1],
                        kernel_size=kernel_size,
                        stride=2,
                        padding=1
                    )
                )
                self.layers.append(nn.ReLU(inplace=True))
                
            elif spec != 'S' and not skip_current:
                self.layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=spec,
                        kernel_size=kernel_size
                    )
                )
                self.layers.append(nn.ReLU(inplace=True))
            
            in_channels = spec

            if not skip_current:
                use_3x3_kernel = not use_3x3_kernel

            skip_current = True if spec == 'S' else False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx % 4 == 3:
                features.append(x)

        return features

    def out_channels(self) -> List[int]:
        return [
            layer.out_channels
            for layer in self.layers[2::4]
        ]