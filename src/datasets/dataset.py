from dataclasses import dataclass
import numpy as np
import os
from PIL import Image
import pandas as pd
from typing import ClassVar, Optional, Any

import hydra
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Data Augmentation
@dataclass
class Transform(object):

    @classmethod    
    def get_transforms(cls, cfg, data):

        if data == 'train':
            return A.Compose([A.Resize(cfg.models.general.image_size,
                                       cfg.models.general.image_size),
                              A.Rotate(limit=15,p=0.8),
                              A.Blur(blur_limit=(1,9),p=0.8),
                              A.RandomBrightnessContrast(brightness_limit=(-0.2,0.4),
                                                         contrast_limit=(-0.2,0.4),
                                                         p=0.8),
                              A.CoarseDropout(min_holes=5,
                                              max_holes=10,
                                              min_width=40,
                                              max_width=40,
                                              min_height=40,
                                              max_height=40,
                                              p=0.8),
                              A.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
                              ToTensorV2()])
        else:
            return A.Compose([A.Resize(cfg.models.general.image_size,
                                       cfg.models.general.image_size),
                              A. Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]),
                              ToTensorV2()])
    
# Dataset Class
@dataclass
class TrainDataset(Dataset):

    cfg: DictConfig
    df: pd.DataFrame
    transform: Optional[Any] = None

    def __post_init__(self):
        self.image_ids = self.df["ID"].values
        self.labels = self.df["target"].values
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.cfg.models.path.image,
                                  self.df["ID"].iloc[idx])
        
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
            image = np.array(image)

        if image is None:
            raise RuntimeError(f"Read fail ! Image path : ({image_path})")

        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

@dataclass
class ValidDataset(Dataset):

    cfg: DictConfig
    df: pd.DataFrame
    transform: Optional[Any] = None

    def __post_init__(self):
        self.image_ids = self.df["ID"].values
        self.labels = self.df["target"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.cfg.models.path.image,
                                  self.df["ID"].iloc[idx])
        
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
            image = np.array(image)

        if image is None:
            raise RuntimeError(f"Read fail ! Image path : ({image_path})")

        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, image_path, image_id

@dataclass
class TestDataset(Dataset):

    cfg: DictConfig
    df: pd.DataFrame
    transform: Optional[Any] = None

    def __post_init__(self):
        self.image_ids = self.df["ID"].values
        self.labels = self.df["target"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.cfg.models.path.image,
                                  self.df["ID"].iloc[idx])
        
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
            image = np.array(image)

        if image is None:
            raise RuntimeError(f"Read fail ! Image path : ({image_path})")

        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, image_path, image_id
