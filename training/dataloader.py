from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from data.voc import load_voc_dataset


def prepare_dataloader(
    split: str,
    batch_size: int,
    local_dir: str = "./data",
    num_workers: int = 0,
) -> DataLoader:

    dataset = load_voc_dataset(
        local_dir=local_dir,
        split=split,
    )

    shuffle = split == "train"
    
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        images = torch.stack(
            [img.float() for img, _ in batch],
            dim=0
        )
        targets = [tgt.float() for _, tgt in batch]

        return images, targets

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def get_voc_dataloaders(
    batch_size: int,
    local_dir: str = "./data",
    num_workers: int = 0,
    splits: List[str] = ("train", "validation", "test"),
) -> Dict[str, DataLoader]:

    return {
        split: prepare_dataloader(
            split=split,
            batch_size=batch_size,
            local_dir=local_dir,
            num_workers=num_workers,
        )
        for split in splits
    }