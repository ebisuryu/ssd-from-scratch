from typing import Tuple

import torch
import torch.nn as nn

from model.backbone.vgg import VGGBackbone
from model.head.extras import ExtraLayers
from model.head.multibox import MultiBoxHead
from model.prior.prior_box import PriorBox
from model.utils import L2Norm
from config import ModelConfig


class SSD(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config
        self.num_classes = config.num_classes
        self.image_size = config.image_size

        self.backbone = VGGBackbone(
            architecture=config.backbone_architecture,
            in_channels=config.backbone_in_channels,
            batch_norm=config.backbone_batch_norm,
        )

        self.l2_norm = L2Norm(
            num_channels=config.l2_norm_channels,
            scale=config.l2_norm_scale,
        )

        self.extras = ExtraLayers(
            architecture=config.extras_architecture,
            in_channels=config.extras_in_channels,
            batch_norm=config.extras_batch_norm,
        )

        self.prior_box = PriorBox(
            image_size=config.image_size,
            feature_map_resolutions=config.feature_map_resolutions,
            box_scales=config.box_scales,
            box_aspect_ratios=config.box_aspect_ratios,
            box_variances=config.box_variances,
            box_clip=config.box_clip
        )
        self.default_boxes = self.prior_box.generate()

        feature_channels = (
            self.backbone.out_channels()
            + self.extras.out_channels()
        )

        self.head = MultiBoxHead(
            in_channels=feature_channels,
            num_classes=config.num_classes,
            num_boxes=config.num_boxes,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)        
        features[0] = self.l2_norm(features[0])
        features.extend(self.extras(features[-1]))
        loc_preds, conf_preds = self.head(features)
        return loc_preds, conf_preds
