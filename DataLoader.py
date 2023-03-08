label_map = {'asteroidan': 0, 'mercurian': 1, 'subterran': 2, 'terran': 3, 'superterran': 4, 'neptunian': 5, 'jovian': 6}
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch.utils.data import DataLoader

class ExoplanetDataset(Dataset):
    def __init__(self, csv_file, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.label_map = label_map
        self.is_test = is_test

        # Compute mean and standard deviation for each feature
        self.mean = torch.tensor(self.df.iloc[:, :3].astype(float).mean().values, dtype=torch.float32)
        self.std = torch.tensor(self.df.iloc[:, :3].astype(float).std().values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get input features (mass, radius, density)
        inputs = torch.tensor(self.df.iloc[idx, :3].astype(float).values, dtype=torch.float32)

        # Normalize input features
        inputs = (inputs - self.mean) / self.std

        if self.is_test:
            return inputs # for new data (planet type still unknown)
        else:
            # Get target from training data (planet type)
            target = self.df.iloc[idx, -1]
            target = label_map[target]
            target = round(float(target))
            target = torch.tensor(target, dtype=torch.long)

            return inputs, target
