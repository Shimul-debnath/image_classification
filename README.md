#Package Installation
pip install torch torchvision 
pip install scikit-learn
pip install matplotlib

# Simple MNIST Classifier with PyTorch


## Features
- **MNIST Dataset**: Used for training and testing the model.
- **Model Architecture**: A simple feed-forward neural network with 2 hidden layers.
- **Training**: The model is trained for 5 epochs using the Adam optimizer and Cross-Entropy loss.
- **Evaluation**: The model's performance is evaluated based on accuracy and loss for both training and validation datasets.
- **Metrics**: Training accuracy, validation accuracy, training loss, validation loss are plotted.
- **Inference Time**: Measures the time taken to predict a batch of 100 samples.
  
## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn


## Results
- The model’s accuracy is displayed after each epoch.
- Training and validation metrics (accuracy & loss) are plotted and saved as `training_validation_plots.png`.
- A classification report is generated showing precision, recall, and F1 scores.


