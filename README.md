Install Necessary Library:

pip install scikit-learn

pip install tensorflow

pip install matplotlib


Data Preparation: The MNIST dataset is normalized by scaling pixel values to the range 0, 1

Model Architecture:
A Flatten layer converts each 28x28 image into a flat vector of 784 values.
Two dense (fully connected) layers provide the model with 128 and 64 neurons respectively.
A final dense layer with 10 neurons (one for each class) uses a softmax activation function to output probabilities.

Model Compilation and Training: The model uses categorical_crossentropy loss and adam optimizer, with 5 epochs.



Input: MNIST dataset

Output: Model Accuracy
