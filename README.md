# PyTorch Exoplanet Classifier

This project uses [PyTorch](https://pytorch.org/) to classify exoplanets based on their mass, radius, and density. The dataset used for this project was provided by [NASA's Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).

This is my first PyTorch and deep learning project. Exoplanet classification predictions made by this model may or may not be correct due to limited training data. This project is solely for educational and demonstration purposes.

### DataLoader

The `DataLoader.py` file defines the `ExoplanetDataset` class, which loads the data from a CSV file and normalizes the input features (planet mass, radius, and density). The `__getitem__` method returns a tuple containing the input features and target label for each data point. If the dataset is used for testing, only the input features are returned.

### Exoplanet Classifier

The `ExoplanetClassifier.py` file defines the `ExoplanetClassifier` class, which is a fully connected neural network model. The number of layers, input size, hidden size, and output size can be configured as input arguments when creating an instance of the model.

The model is trained using the `train_loader` and `val_loader` created from the `ExoplanetDataset` class. The training process includes the following steps:

* Forward pass through the model to obtain the predicted labels

* Calculate the loss between the predicted and true labels using cross entropy loss

* Backward pass through the model to calculate the gradients of the loss with respect to the model parameters

* Update the model parameters using the optimizer (Adam optimizer is used in this project)


The model is evaluated on the validation set after each epoch of training. The validation process includes the following steps:

* Forward pass through the model to obtain the predicted labels

* Calculate the loss between the predicted and true labels using cross entropy loss

* Calculate the accuracy of the model by comparing the predicted labels with the true labels

The trained model is saved to a file named `exoplanet_classifier.pth` using the `torch.save` function.

## How to Use

To run the exoplanet classifier on a local machine, follow these steps:

1. Clone the repository to your local machine using the following command:

	`git clone https://github.com/jarvisar/exoplanet-classifier.git`
    
2. Change directory into the root folder of the cloned repository:

	`cd exoplanet-classifier`
    
3. Configure the training set `exoplanet_data.csv` and new data `new_data.csv`. 
    
4. Run the main.py file to start the classifier:

	`python main.py`
   
   
   
### Known Issues & Limitations

The training data used in this project is limited due to the difficulty of manually gathering exoplanet data and correctly classifying it. As a result, the model's performance may be limited by the small dataset size.

This project is intended to serve as a demonstration of PyTorch and deep learning techniques for exoplanet classification. It was undertaken as a personal interest project to develop my skills in PyTorch and deep learning. The project demonstrates proficiency in Python and my ability to implement a fully connected neural network model for exoplanet classification. Even though the training dataset used in this project is limited, the techniques used and the models developed in this project provide a strong foundation for further exploration and analysis in the future.
     
<br>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png"/>
</p>
