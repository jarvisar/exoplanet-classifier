import torch
import numpy as np
from DataLoader import ExoplanetDataset
from ExoplanetClassifier import ExoplanetClassifier
from torch.utils.data import Dataset, DataLoader

label_map = {'asteroidan': 0, 'mercurian': 1, 'subterran': 2, 'terran': 3, 'superterran': 4, 'neptunian': 5, 'jovian': 6}

# Define model
model = ExoplanetClassifier(num_layers=4)

while True:
    # Accept user input for planet parameters
    input_str = input("Enter planet parameters in the format 'mass,radius,density' or type 'exit' to quit: ")
    if input_str == 'exit':
        break
    input_list = input_str.split(',')
    input_arr = np.array(input_list, dtype=np.float32)

    # Convert input array to PyTorch tensor
    inputs = torch.from_numpy(input_arr).unsqueeze(0)

    # Use the pre-trained model to make predictions
    with torch.no_grad():
        predictions = model(inputs)
        predicted_classes = predictions.argmax(dim=1)
        print(list(label_map.keys())[list(label_map.values()).index(predicted_classes.item())])


