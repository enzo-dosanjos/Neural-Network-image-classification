# Neural Network Image Classifier in C++

## Overview
This project implements a simple neural network for image classification tasks, written entirely in C++ without external libraries. The neural network is designed to recognize handwritten digits (0-9) from any images (it will resize and convert to greyscale image) such as those in the MNIST dataset. The project includes functions for training the model, testing it, and making predictions on new images. The neural network uses:

- Forward propagation for computing predictions.
- Backpropagation to calculate gradients.
- Gradient descent for optimizing weights and biases.

The dataset I used to train the model contained in the saved_model directory is the MNIST dataset in jpeg format which can be found here: https://github.com/teavanist/MNIST-JPG.

---

## How to Use:
### 1. Compilation and Execution:
First, uncomment the function you want to run in the main.cpp file. Then, to compile the project, simply run "make" in the terminal. Then run "./NN".

### 2. Train the Model:
To train the model, use the trainModel function and a dataset. Provide the following parameters:

- model_path: Path to save the trained model (weights and biases).
- data_csv_path: Path to the CSV file containing the training data. (A file is provided to create the csv file automatically in utilities)
- data_size: Number of training samples.
- batch_size: Number of samples per training iteration.
- input_size: Size of the input layer (e.g. 28 * 28 for MNIST images).
- epochs: Number of training iterations over the dataset.
- learning_rate: The learning rate for gradient descent.

(Personalisation of the neural network will come later)

### 3. Improve the Model
Fine-tune an existing model with additional training data or simply improve it's accuracy by running it again through the same dataset usng the improveModel function. Provide the same parameters as the trainModel function.

### 4. Test the Model:
Evaluate the accuracy of a trained model using the testModel function and a test dataset. Provide the following parameters:

    model_path: Path to the trained model to load.
    data_csv_path: Path to the CSV file containing the test data.
    data_size: Number of test samples.
    input_size: Size of the input layer.

### 4. Make Predictions:
Use a trained model to predict the label of a single input image using the predict function. Provide the following parameters:

    model_path: Path to the trained model to load.
    img_path: Path to the image you want to classify.
    input_size: Size of the input layer.

---

## Future Improvements

- Add support for additional activation functions (e.g., sigmoid, tanh).
- Extend to multi-layer architectures with more hidden layers.
- Improve performance with parallelization or optimized matrix operations.
- Find how to use the performance of the computer's GPU