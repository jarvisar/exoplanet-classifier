label_map = {'asteroidan': 0, 'mercurian': 1, 'subterran': 2, 'terran': 3, 'superterran': 4, 'neptunian': 5, 'jovian': 6}
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class ExoplanetDataset(Dataset):
    def __init__(self, csv_file, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.label_map = label_map
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get input features (mass, radius, density)
        inputs = torch.tensor(self.df.iloc[idx, :3].astype(float).values, dtype=torch.float32)

        if self.is_test:
            return inputs
        else:
            # Get target (planet type)
            target = self.df.iloc[idx, -1]
            target = label_map[target]
            target = round(float(target))
            target = torch.tensor(target, dtype=torch.long)

            return inputs, target

dataset = ExoplanetDataset('exoplanet_data.csv', label_map)
