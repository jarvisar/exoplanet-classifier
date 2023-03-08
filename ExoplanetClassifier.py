import torch.nn as nn
import torch.optim as optim
from DataLoader import ExoplanetDataset
import torch

label_map = {'asteroidan': 0, 'mercurian': 1, 'subterran': 2, 'terran': 3, 'superterran': 4, 'neptunian': 5, 'jovian': 6}

# Define fully connected neural network model
class ExoplanetClassifier(nn.Module):
    def __init__(self):
        super(ExoplanetClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 7)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load training data
train_loader = ExoplanetDataset('exoplanet_data.csv')

# Define model, loss function and optimizer
model = ExoplanetClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # Get inputs and targets from data loader
        inputs, targets = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass through the model
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Print running loss every 1000 batches
        running_loss += loss.item()
        if i % 1000 == 999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0


# Save the trained model
torch.save(model.state_dict(), 'exoplanet_classifier.pth')
