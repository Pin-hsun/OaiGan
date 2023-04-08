import os
import glob
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Find all subject IDs
        self.subjects = sorted(set(os.path.basename(file_path)
                                   .replace('_' + os.path.basename(file_path).split('_')[-1], '')
                                   for file_path in glob.glob(os.path.join(data_dir, '*.tif'))))

        # Group file paths by subject ID
        self.file_groups = {subject: sorted(glob.glob(os.path.join(data_dir, '{}_*.tif'.format(subject)))) for subject in self.subjects}

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        file_paths = self.file_groups[subject_id]

        # Load and stack image slices
        slices = [np.array(Image.open(file_path)) for file_path in file_paths]
        x = np.stack(slices, axis=0)

        # Apply transforms
        if self.transform is not None:
            x = self.transform(image=x)['image']

        # Convert to PyTorch tensor
        x = torch.from_numpy(x).float()

        return x


# Define Albumentations transformations
transform = A.Compose([
    A.RandomCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15),
    A.Normalize()
])

# List of data directories
root = '/media/ExtHDD01/Dataset/paired_images/womac4/full'
data_dirs = [os.path.join(root, d) for d in ['ap', 'bp']]

# Concatenate datasets from all directories
d = CustomDataset(data_dirs[0], transform=transform)
