import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from DataLoader import ExoplanetDataset
from sklearn.model_selection import train_test_split

label_map = {'asteroidan': 0, 'mercurian': 1, 'subterran': 2, 'terran': 3, 'superterran': 4, 'neptunian': 5, 'jovian': 6}

# Define fully connected neural network model


class ExoplanetClassifier(nn.Module):
    def __init__(self, num_layers=3, input_size=3, hidden_size=64, output_size=7):
        super(ExoplanetClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-2)])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x


# Load data
exoplanet_data = ExoplanetDataset('exoplanet_data.csv')

# Split data into training and validation sets
train_data, val_data = train_test_split(exoplanet_data, test_size=0.2, random_state=42)

# Create data loaders for training and validation sets
train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = data.DataLoader(val_data, batch_size=32, shuffle=False)

# Define model, loss function and optimizer
model = ExoplanetClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    running_loss = 0.0
    
    # Train model on training set
    model.train()
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
            print('[%d, %5d] training loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

    # Evaluate model on validation set
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for data in val_loader:
            # Get inputs and targets from data loader
            inputs, targets = data

            # Forward pass through the model
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Calculate number of correct predictions
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == targets).sum().item()

    # Print validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_data)
    print('[%d] validation loss: %.3f, validation accuracy: %.3f' % (epoch + 1, val_loss, val_acc))

# Save the trained model
torch.save(model.state_dict(), 'exoplanet_classifier.pth')