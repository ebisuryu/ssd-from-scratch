```
ssd-from-scratch/
│
├── configs/
│   ├── model/
│   │   ├── ssd300.yaml              # SSD300 model configuration
│   │   ├── ssd512.yaml              # SSD512 model configuration
│   │   └── backbone_vgg16.yaml      # VGG16 backbone settings
│   │
│   ├── data/
│   │   ├── voc.yaml                 # Pascal VOC dataset config
│   │   └── coco.yaml                # COCO dataset config
│   │
│   ├── train.yaml                   # Training hyperparameters
│   └── eval.yaml                    # Evaluation configuration
│
├── ssd/                             # Core SSD library
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ssd.py                   # Main SSD model
│   │   │
│   │   ├── backbone/
│   │   │   ├── __init__.py
│   │   │   ├── vgg.py               # VGG backbone implementation
│   │   │   └── base.py              # Backbone interface / abstract class
│   │   │
│   │   ├── head/
│   │   │   ├── __init__.py
│   │   │   ├── multibox.py          # Localization & classification heads
│   │   │   └── extras.py            # Extra feature layers for SSD
│   │   │
│   │   └── prior/
│   │       ├── __init__.py
│   │       └── prior_box.py         # Default boxes (anchors) generation
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets/
│   │   │   ├── voc.py               # Pascal VOC dataset
│   │   │   ├── coco.py              # COCO dataset
│   │   │   └── base.py              # Base dataset class
│   │   │
│   │   ├── transforms/
│   │   │   ├── __init__.py
│   │   │   ├── augmentations.py     # SSD-style data augmentation
│   │   │   └── compose.py           # Transform composition
│   │   │
│   │   ├── collate.py               # Custom collate function
│   │   └── sampler.py               # Dataset sampler
│   │
│   ├── loss/
│   │   ├── __init__.py
│   │   ├── multibox_loss.py         # Smooth L1 + Cross Entropy loss
│   │   └── matcher.py               # IoU-based matching (GT ↔ priors)
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop
│   │   ├── evaluator.py             # Evaluation & mAP computation
│   │   └── inference.py             # Inference pipeline
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── map.py                   # Mean Average Precision (mAP)
│   │   └── iou.py                   # IoU computation utilities
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── box_ops.py               # Box encode / decode utilities
│   │   ├── nms.py                   # Non-Maximum Suppression
│   │   ├── visualization.py         # Bounding box visualization
│   │   ├── logging.py               # Logging utilities
│   │   ├── checkpoint.py            # Checkpoint saving / loading
│   │   └── seed.py                  # Reproducibility utilities
│
├── scripts/
│   ├── prepare_voc.py               # Prepare Pascal VOC dataset
│   ├── prepare_coco.py              # Prepare COCO dataset
│   └── download_weights.sh          # Download pretrained weights
│
├── weights/
│   └── README.md                    # Pretrained weights description
│
├── outputs/
│   ├── logs/                        # Training logs
│   ├── checkpoints/                # Saved checkpoints
│   └── predictions/                # Inference outputs
│
├── tests/
│   ├── test_prior_box.py            # Unit tests for prior boxes
│   ├── test_iou.py                  # Unit tests for IoU
│   └── test_loss.py                 # Unit tests for loss functions
│
├── train.py                         # Training entrypoint
├── evaluate.py                      # Evaluation entrypoint
├── inference.py                     # Inference entrypoint
│
├── README.md
└── requirements.txt
```