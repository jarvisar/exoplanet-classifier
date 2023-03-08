import torch
import numpy as np
from DataLoader import ExoplanetDataset
from ExoplanetClassifier import ExoplanetClassifier
from torch.utils.data import Dataset, DataLoader

label_map = {'asteroidan': 0, 'mercurian': 1, 'subterran': 2, 'terran': 3, 'superterran': 4, 'neptunian': 5, 'jovian': 6}

# Convert the new data into PyTorch dataset and dataloader
new_dataset = ExoplanetDataset('new_data.csv', is_test=True)
new_dataloader = DataLoader(new_dataset, batch_size=1)
model = ExoplanetClassifier(num_layers=4)
# Use the pre-trained model to make predictions
with torch.no_grad():
    for inputs in new_dataloader:
        predictions = model(inputs)
        predicted_classes = predictions.argmax(dim=1)
        print(list(label_map.keys())[list(label_map.values()).index(predicted_classes.item())])
