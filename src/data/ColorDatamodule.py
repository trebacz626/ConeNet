# create datamodule
import torch
import lightning as pl

from src.data.ColorDataset import ColorDataset
import numpy as np
from sklearn.model_selection import train_test_split

class ColorDatamodule(pl.LightningDataModule):
    def __init__(self, root_folder, train_transformations, valid_transformations,
                 batch_size=32, num_workers=0, test_size=0.2):
        super().__init__()
        self.root_folder = root_folder
        self.train_transformations = train_transformations
        self.valid_transformations = valid_transformations
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = np.load(root_folder + '/data.npy')
        self.labels = np.load(root_folder + '/labels.npy')
        self.test_size = test_size

    def setup(self, stage=None):
        # split data into train and validation
        self.train_data, self.valid_data, self.train_labels, self.valid_labels = train_test_split(self.data, self.labels, test_size=self.test_size, random_state=42)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            ColorDataset(self.root_folder, self.train_data, self.train_labels, self.train_transformations),
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            ColorDataset(self.root_folder, self.valid_data, self.valid_labels, self.valid_transformations),
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()
