#Import Libraries
import os
import cv2
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

import albumentations as A
from albumentations import (Compose, OneOf, Normalize,
                            CenterCrop, Blur, Resize, 
                            RandomResizedCrop, RandomCrop, 
                            HorizontalFlip, VerticalFlip, 
                            RandomBrightness, RandomContrast,
                            RandomBrightnessContrast, RandomRotate90,
                            ShiftScaleRotate, Transpose, 
                            HueSaturationValue, CoarseDropout, GridDropout)

from albumentations.pytorch import ToTensorV2

from utils.Configuration import CFG
from utils.Parser import get_args

#DataAugmentation
def get_transforms(data):

    opt = get_args()

    if data == 'train':
        return Compose([A.Resize(opt.image_size, opt.image_size),
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
        return Compose([Resize(opt.image_size, opt.image_size),
                        Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
                        ToTensorV2()])

class TrainDataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.image_ids = df["ID"].values
        self.labels = df["target"].values
        self.transform = transform
        self.opt = get_args()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.opt.image_path, self.df["ID"].iloc[idx])
        
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
            image = np.array(image)

        if image is None:
            raise RuntimeError(f"read fail '{image_path}'")

        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


class TestDataset(Dataset):

    opt = get_args()

    def __init__(self, df, transform=None):
        self.df = df
        self.image_ids = df["ID"].values
        self.labels = df["target"].values
        self.transform = transform
        self.opt = get_args()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.opt.image_path, self.df["ID"].iloc[idx])
        
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
            image = np.array(image)

        if image is None:
            raise RuntimeError(f"read fail '{image_path}'")

        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, image_path, image_id

