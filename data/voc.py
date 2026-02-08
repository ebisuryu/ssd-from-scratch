from typing import List, Tuple, Literal, Optional, Callable
import os


import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

from data.transform import SSDAugmentation, VOCAnnotationTransform


class VOCDataset(Dataset):

    def __init__(
        self,
        root: str,
        splits: List[Tuple[str, str]] = [("2007", "train"), ("2012", "train")],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.root = root
        self.splits = splits

        self.transform = transform
        self.target_transform = target_transform or VOCAnnotationTransform()

        self.annotation_template = os.path.join("%s", "annotations", "%s.xml")
        self.image_template = os.path.join("%s", "images", "%s.jpg")

        self.samples: List[Tuple[str, str]] = self._load_image_ids()

    def _load_image_ids(self) -> List[Tuple[str, str]]:
        samples = []
        voc_root = os.path.join(self.root, "voc")

        for year, split in self.splits:
            split_file = os.path.join(voc_root, year, "main", f"{split}.txt")
            with open(split_file, "r") as f:
                for line in f:
                    samples.append((f"{voc_root}/{year}", line.strip()))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        root_path, image_id = self.samples[index]

        image_path = self.image_template % (root_path, image_id)
        annotation_path = self.annotation_template % (root_path, image_id)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        annotation = ET.parse(annotation_path).getroot()
        height, width = image.shape[:2]

        target = self.target_transform(annotation, width, height)
        target = np.array(target, dtype=np.float32)

        if self.transform is not None:
            boxes = target[:, :4]
            labels = target[:, 4]
            image, boxes, labels = self.transform(image, boxes, labels)
            target = np.hstack([boxes, labels[:, None]])

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        target = torch.from_numpy(target)

        return image, target


def load_voc_dataset(
    local_dir: str,
    split: Literal["train", "validation", "test"],
) -> Dataset:
    split_mappings = {
        "train": [("2007", "train"), ("2012", "train")],
        "validation": [("2007", "val"), ("2012", "val")],
        "test": [("2007", "test")]
    }
    splits = split_mappings[split]
    dataset = VOCDataset(
        root=local_dir,
        splits=splits,
        transform=SSDAugmentation(),
        target_transform=VOCAnnotationTransform()
    )
    return dataset