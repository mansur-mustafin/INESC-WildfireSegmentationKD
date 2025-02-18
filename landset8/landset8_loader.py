import os
import random

import pickle
import rasterio
import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import Dataset

MAX_PIXEL_VALUE = 65535


class Landset8Dataset(Dataset):

    def __init__(self, images_csv, dataset_dir, mask, augment=False):
        """
        Args:
            mask: Kumar, Schroeder, Murphy, intersections, voting
        """
        self.data = pd.read_csv(images_csv, skiprows=1, header=None)[0].tolist()
        self.dataset_dir = dataset_dir
        self.mask = mask
        self.seed = 1234
        with open(dataset_dir+"dataset.pkl", "rb") as file:
            data = pickle.load(file)
        self.df = {fp.split('/')[-1]: row for fp, row in data.set_index('full_path').iterrows()}

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
        image_path, mask_path = self._generate_path(image_filename)

        image_path = os.path.join(self.dataset_dir, image_path)
        mask_path = os.path.join(self.dataset_dir, mask_path)

        with rasterio.open(image_path) as src:
            image = src.read((7, 6, 2)).astype(np.float32) / MAX_PIXEL_VALUE
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

    def _generate_path(self, image_filename):
        """
        Generate mask filename by inserting the mask
        """
        row = self.df.get(image_filename, "File not found")
        column = 'masks_'+self.mask
        return row.name, row[column]
