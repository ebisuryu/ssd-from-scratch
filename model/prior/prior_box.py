from typing import Dict, List
from itertools import product
from math import sqrt

import torch


class PriorBox:

    def __init__(
        self, 
        image_size: int,
        feature_map_resolutions: List[int],
        box_scales: List[float],
        box_aspect_ratios: List[List[float]],
        box_variances: List[float],
        box_clip: bool
    ) -> None:
        self.image_size = image_size
        self.feature_map_resolutions = feature_map_resolutions
        self.box_scales = box_scales
        self.box_aspect_ratios = box_aspect_ratios
        self.box_variances = box_variances
        self.box_clip = box_clip

        for box_variance in self.box_variances:
            if box_variance <= 0:
                raise ValueError("Variance values must be greater than 0")

    def generate(self) -> torch.Tensor:
        priors = []

        for idx, feature_map_resolution in enumerate(self.feature_map_resolutions):

            box_scale = self.box_scales[idx]
            next_box_scale = (
                self.box_scales[idx + 1]
                if idx + 1 < len(self.box_scales)
                else 1.05
            )

            for i, j in product(range(feature_map_resolution), repeat=2):
                cx = (j + 0.5) / feature_map_resolution
                cy = (i + 0.5) / feature_map_resolution

                size = sqrt(box_scale * next_box_scale)
                priors.extend([cx, cy, size, size])
                
                for ratio in self.box_aspect_ratios[idx]:
                    priors.extend([
                        cx,
                        cy,
                        box_scale * sqrt(ratio),
                        box_scale / sqrt(ratio),
                    ])

        output = torch.tensor(priors).view(-1, 4)

        if self.box_clip:
            output.clamp_(min=0.0, max=1.0)

        return output