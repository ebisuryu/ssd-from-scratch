from typing import List, Dict, Type, Any, Tuple
from dataclasses import dataclass, field
import yaml


@dataclass
class DataConfig:
    
    labels: List[str]
    background_label: int


@dataclass
class ModelConfig:
    
    num_classes: int
    
    image_size: int
    feature_map_resolutions: List[int]
    box_scales: List[float]
    box_aspect_ratios: List[List[float]]
    box_variances: List[float]
    box_clip: bool
    
    backbone_architecture: List = field(default_factory=list)
    backbone_in_channels: int = 3
    backbone_batch_norm: bool = False

    l2_norm_channels: int = 512
    l2_norm_scale: float = 20.0

    extras_architecture: List = field(default_factory=list)
    extras_in_channels: int = 1024
    extras_batch_norm: bool = False

    num_boxes: List[int] = field(default_factory=list)
    

@dataclass
class TrainConfig:

    batch_size: int
    epochs: int
    grad_accumulation_steps: int
    gradient_clip_norm: float
    
    optimizer: str
    learning_rate: float
    gamma: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    
    monitor_metric: str
    monitor_mode: str
    checkpoint_dir: str
    save_best_only: bool
    
    device: str
    num_workers: int
    seed: int


@dataclass
class EvaluateConfig:
    pass


def load_config_from_yaml(config_type: str, file_path: str) -> Any:
    config_registry: Dict[str, Type] = {
        "data": DataConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "evaluate": EvaluateConfig
    }

    if config_type not in config_registry:
        raise ValueError(
            f"Unknown config type '{config_type}'. "
            f"Available types: {list(config_registry.keys())}"
        )

    config_cls = config_registry[config_type]

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        return config_cls(**data)
    except TypeError as e:
        raise ValueError(
            f"Invalid YAML for config type '{config_type}' at {file_path}"
        ) from e