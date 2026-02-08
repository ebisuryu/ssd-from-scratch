import torch
import torch.nn as nn


class L2Norm(nn.Module):

    def __init__(self, num_channels: int, scale: float) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.scale = scale
        self.eps = 1e-10

        self.scale_weights = nn.Parameter(
            torch.empty(num_channels)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.scale_weights, self.scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.sqrt(features.pow(2).sum(dim=1, keepdim=True)) + self.eps

        normalized_features = features / l2_norm
        scaled_features = self.scale_weights.view(1, self.num_channels, 1, 1) * normalized_features

        return scaled_features
