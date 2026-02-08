from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_bboxes(
    rgb_image: np.ndarray, 
    bboxes: torch.Tensor, 
    labels: List[str], 
    conf_threshold: float = 0.6
) -> None:
    plt.imshow(rgb_image)
    colors = plt.cm.hsv(np.linspace(0, 1, bboxes.size(1)))
    scale = torch.tensor(rgb_image.shape[1::-1]).repeat(2)
    ax = plt.gca()
    
    for i in range(1, bboxes.size(1)):  
        j = 0
        while bboxes[0, i, j, 0] >= conf_threshold:
            score = bboxes[0, i, j, 0].item()
            xmin, ymin, xmax, ymax = bboxes[0, i, j, 1:] * scale
            ax.add_patch(plt.Rectangle(
                (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1,
                fill=False, edgecolor=colors[i], linewidth=2
            ))
            ax.text(xmin, ymin, f"{labels[i-1]}: {score:.2f}",
                    bbox=dict(facecolor=colors[i], alpha=0.5))
            j += 1

    plt.axis("off")
    plt.show()