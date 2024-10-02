import os
import random

import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader


class Sen2FireDataset(Dataset):

    def __init__(self, data_dir, augment=False):
        self.file_list = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.npz')]
        self.seed = 1234

        random.seed(self.seed)

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
        return len(self.file_list)
    
    def add_data_dir(self, data_dir):
        aditional_file_list = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.npz')]
        self.file_list += aditional_file_list

    def __getitem__(self, index):
        file_path = self.file_list[index]
        data = np.load(file_path)

        images = data['image'].astype(np.float32) / 10000  # Normalize if needed
        aerosol = data['aerosol']
        label = data['label']

        # Stack images and aerosol
        combined_image = np.concatenate([images, aerosol[np.newaxis, ...]], axis=0)

        # Transpose to HxWxC for Albumentations
        combined_image = combined_image.transpose(1, 2, 0)
        label = label

        # Apply augmentation
        augmented = self.transform(image=combined_image, mask=label)
        combined_image = augmented['image']
        label = augmented['mask']

        # Transpose back to CxHxW
        combined_image = combined_image.transpose(2, 0, 1)

        # Convert to tensors
        combined_image = torch.tensor(combined_image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return combined_image, label
