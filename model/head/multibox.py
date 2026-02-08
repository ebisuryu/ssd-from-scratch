from typing import List, Tuple
import torch
import torch.nn as nn


class MultiBoxHead(nn.Module):

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        num_boxes: List[int],
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        for channels, boxes in zip(in_channels, num_boxes):
            self.loc_layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=boxes * 4,
                    kernel_size=3,
                    padding=1,
                )
            )
            self.conf_layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=boxes * num_classes,
                    kernel_size=3,
                    padding=1,
                )
            )

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        loc_preds = []
        conf_preds = []

        for x, loc_layer, conf_layer in zip(
            features, self.loc_layers, self.conf_layers
        ):
            loc = loc_layer(x)
            conf = conf_layer(x)

            loc = loc.permute(0, 2, 3, 1).contiguous()
            conf = conf.permute(0, 2, 3, 1).contiguous()

            loc_preds.append(loc.view(loc.size(0), -1, 4))
            conf_preds.append(conf.view(conf.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, dim=1)
        conf_preds = torch.cat(conf_preds, dim=1)

        return loc_preds, conf_preds