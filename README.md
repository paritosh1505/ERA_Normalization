# Code Repository

This repository contains code for a deep learning model using PyTorch. The code is divided into multiple files and provides functionality for training and evaluating the model.

## Code 1: Model Definition

__File: model.py__

This file contains the definition of the neural network model Net class. The model architecture consists of multiple convolutional and normalization layers. The choice of normalization (batch normalization or group normalization) can be specified using the norm_value parameter.

The forward method defines the forward pass of the model. The  get_norm_layer  method is used to select the appropriate normalization layer based on the given  norm_value .

The  model_summary  function can be used to print a summary of the model architecture and the size of the input.

## Code 2: Helper Functions and Data Loading

File:  helpers.py 

This file contains various helper functions for data loading, visualization, and augmentation.

The  load_data  function loads the CIFAR-10 dataset and applies data transformations using the Albumentations library for image augmentation. It returns the train and test data loaders.

The  plotings  class provides a method for plotting images from the dataset.

## Code 3: Training and Evaluation

File:  performance.py 

This file contains the  Performance  class, which is responsible for training and evaluating the model.

The  train  method performs the training loop, iterating over the train data loader, calculating the loss, and updating the model parameters using backpropagation. It also supports L1 regularization.

The  test  method evaluates the trained model on the test data and calculates the accuracy and loss.

The  scores  function returns the training and testing loss and accuracy values.

## Usage

To use this code, follow these steps:

1. Install the required dependencies listed in the  requirements.txt  file.

2. Modify the parameters in the  main.py  file as needed.

3. Run the  main.py  script to train and evaluate the model.


