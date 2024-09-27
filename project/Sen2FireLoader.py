import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Sen2FireDataset(Dataset):

    def __init__(self, data_dir):
        self.file_list = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.npz')]

    def __len__(self):
        return len(self.file_list)
    
    def add_data_dir(self, data_dir):
        aditional_file_list = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.npz')]
        self.file_list += aditional_file_list

    def __getitem__(self, index):
        file_path = self.file_list[index]

        data = np.load(file_path)

        images = data['image'].astype(np.float32) / 10000
        aerosol = data['aerosol']
        label = data['label']

        images = torch.tensor(images, dtype=torch.float32)
        aerosol = torch.tensor(aerosol, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(label, dtype=torch.long)

        combined_image = torch.cat((images, aerosol), dim=0)

        return combined_image, label
