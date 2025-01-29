import os
import random

import rasterio
import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader

MAX_PIXEL_VALUE = 65535


class Landset8Dataset(Dataset):

    def __init__(self, images_csv, images_dir, masks_dir, mask_postfix, augment=False):
        """
        Args:
            mask_postfix (str): Postfix added to image filenames to get mask filenames.
            In this example: LC08_L1TP_119031_20200807_20200807_01_RT_p00273.tif,LC08_L1TP_119031_20200807_20200807_01_RT_Schroeder_p00273.tif
            mask_postfix = _Schroeder
        """
        self.data = pd.read_csv(images_csv, skiprows=1, header=None)[0].tolist()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mask_postfix = mask_postfix
        self.seed = 1234

        random.seed(self.seed)
        np.random.seed(self.seed)

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ])
        else:
            self.transform = A.Compose([])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_filename = self.data[index].strip()
        mask_filename = self._generate_mask_filename(image_filename)

        image_path = os.path.join(self.images_dir, image_filename)
        mask_path = os.path.join(self.masks_dir, mask_filename)

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32) / MAX_PIXEL_VALUE
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        image = image.transpose(1, 2, 0)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        image = image.transpose(2, 0, 1)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

    def _generate_mask_filename(self, image_filename):
        """
        Generate mask filename by inserting the mask_postfix
        before `_pXXXXX` in the image filename.
        """
        base_name, extension = image_filename.rsplit(".", 1)
        parts = base_name.split("_p")  # Split before `_pXXXXX`
        mask_filename = f"{parts[0]}{self.mask_postfix}_p{parts[1]}.{extension}"
        return mask_filename
