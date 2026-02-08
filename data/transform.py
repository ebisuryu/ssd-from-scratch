from typing import List

import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

import types
import cv2
import torch
import random
import numpy as np
from torchvision import transforms

from utils import jaccard


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]


class Compose:
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, boxes=None, labels=None):
        for transform in self.transforms:
            image, boxes, labels = transform(image, boxes, labels)
        return image, boxes, labels
    
    
class Lambda:
    
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd
    
    def __call__(self, image, boxes=None, labels=None):
        return self.lambd(image, boxes, labels)


class ToFloat32:
    
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class MeanNormalizer:
    
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)
    
    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class DenormalizeBoundingBoxes:
    
    def __call__(self, image, boxes=None, labels=None):
        height, width, channel = image.shape
        boxes[:, 0] = boxes[:, 0] * width
        boxes[:, 2] = boxes[:, 2] * width
        boxes[:, 1] = boxes[:, 1] * height
        boxes[:, 3] = boxes[:, 3] * height
        return image, boxes, labels
    
    
class NormalizeBoundingBoxes:
    
    def __call__(self, image, boxes=None, labels=None):
        height, width, channel = image.shape
        boxes[:, 0] = boxes[:, 0] / width
        boxes[:, 2] = boxes[:, 2] / width
        boxes[:, 1] = boxes[:, 1] / height
        boxes[:, 3] = boxes[:, 3] / height
        return image, boxes, labels
    
    
class Resize:
    
    def __init__(self, size=300):
        self.size = size
        
    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels
    
    
class RandomSaturation:
    
    def __init__(self, lower=0.5, upper=1):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 1] = image[:, :, 1] * np.random.uniform(self.lower, self.upper)
        return image, boxes, labels


class RandomHue:
    
    def __init__(self, delta=18.0):
        self.delta = delta
    
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class ChannelSwapper:

    def __init__(self, swap):
        self.swap = swap
    
    def __call__(self, image):
        image = image[:, :, self.swap]
        return image


class RandomLightNoise:
    
    def __init__(self):
        self.perms = (
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0)
        )
    
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = ChannelSwapper(swap)
            image = shuffle(image)
        return image, boxes, labels


class ColorTransformer:
    
    def __init__(self, current='BGR', transform='HSV'):
        self.current = current
        self.transform = transform
        
    def __call__(self, image, boxes=None, labels=None):
        if (self.current == 'BGR') and (self.transform == 'HSV'):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif (self.current == 'HSV') and (self.transform == 'BGR'):
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast:
    
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
    
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image = image * alpha
        return image, boxes, labels
    
    
class RandomBrightness:
    
    def __init__(self, delta=32):
        self.delta = delta
    
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels
    
    
class RandomSampleCrop:
    
    def __init__(self):
        self.options = [
            None,
            (0.1, None), (0.3, None), (0.7, None), (0.9, None),
            (None, None)
        ]
    
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            mode = random.choice(seq=self.options)
            if mode is None:
                return image, boxes, labels
            
            min_iou, max_iou = mode
            min_iou = float('-inf') if min_iou is None else min_iou
            max_iou = float('inf') if max_iou is None else max_iou
            
            for _ in range(50):
                current_image = image
                
                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)
                
                if (h / w < 0.5) or (h / w > 2):
                    continue
                
                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)
                
                rectangular = np.array([int(left), int(top), int(left + w), int(top + h)])
                
                overlap = jaccard(boxes, rectangular)
                
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                
                current_image = current_image[rectangular[1]:rectangular[3], rectangular[0]:rectangular[2], :]
                
                center_points = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                
                mask_1 = (rectangular[0] < center_points[:, 0]) * (rectangular[1] < center_points[:, 1])
                mask_2 = (rectangular[2] > center_points[:, 0]) * (rectangular[3] > center_points[:, 1])

                mask = mask_1 * mask_2
                
                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]
                
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rectangular[:2])
                current_boxes[:, :2] -= rectangular[:2]
                
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rectangular[2:])
                current_boxes[:, 2:] -= rectangular[:2]
                
                return current_image, current_boxes, current_labels
            
            
class RandomExpand:
    
    def __init__(self, mean):
        self.mean = mean
    
    def __call__(self, image, boxes, labels):
        if np.random.randint(2):
            return image, boxes, labels
        
        height, width, channel = image.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)
        
        expanded_image = np.zeros(
            shape=(int(height * ratio), int(width * ratio), channel),
            dtype=image.dtype
        )
        expanded_image[:, :, :] = self.mean
        expanded_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expanded_image
        
        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        
        return image, boxes, labels


class RandomHorizontalFlip:
    
    def __call__(self, image, boxes, labels):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels


class RandomPhotometricAugmentation:
    
    def __init__(self):
        self.photometric_transforms = [
            RandomContrast(),
            ColorTransformer(current='BGR', transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ColorTransformer(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightNoise()
    
    def __call__(self, image, boxes, labels):
        image = image.copy()
        image, boxes, labels = self.rand_brightness(image, boxes, labels)
        if np.random.randint(2):
            photometric_pipeline = Compose(self.photometric_transforms[:-1])
        else:
            photometric_pipeline = Compose(self.photometric_transforms[1:])
        image, boxes, labels = photometric_pipeline (image, boxes, labels)
        return self.rand_light_noise(image, boxes, labels)


class SSDAugmentation:
    
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment_pipeline = Compose(
            transforms=[
                ToFloat32(), 
                DenormalizeBoundingBoxes(),
                RandomPhotometricAugmentation(), 
                RandomExpand(mean=self.mean),
                # RandomSampleCrop(), 
                RandomHorizontalFlip(),
                NormalizeBoundingBoxes(), 
                Resize(size=self.size),
                MeanNormalizer(mean=self.mean)
            ]
        )
    
    def __call__(self, image, boxes, labels):
        return self.augment_pipeline(image, boxes, labels)
    

class VOCAnnotationTransform:

    def __init__(
        self,
        keep_difficult: bool = False,
    ):
        self.class_to_idx = {
            name: idx for idx, name in enumerate(VOC_CLASSES)
        }
        self.keep_difficult = keep_difficult

    def __call__(
        self,
        annotation: ET.Element,
        image_width: int,
        image_height: int,
    ) -> List[List[float]]:
        targets: List[List[float]] = []

        for obj in annotation.iter("object"):
            difficult = int(obj.findtext("difficult", default="0")) == 1
            if difficult and not self.keep_difficult:
                continue

            class_name = obj.findtext("name").strip().lower()
            if class_name not in self.class_to_idx:
                continue

            bbox = obj.find("bndbox")
            if bbox is None:
                continue

            xmin = (int(bbox.findtext("xmin")) - 1) / image_width
            ymin = (int(bbox.findtext("ymin")) - 1) / image_height
            xmax = (int(bbox.findtext("xmax")) - 1) / image_width
            ymax = (int(bbox.findtext("ymax")) - 1) / image_height

            label = self.class_to_idx[class_name]
            targets.append([xmin, ymin, xmax, ymax, label])

        return targets