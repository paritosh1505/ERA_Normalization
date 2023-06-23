# Code Repository

This repository contains code for a deep learning model using PyTorch. The code is divided into multiple files and provides functionality for training and evaluating the model.

## Code 1: Model Definition

__File: model.py__

This file contains the definition of the neural network model Net class. The model architecture consists of multiple convolutional and normalization layers. The choice of normalization (batch normalization or group normalization) can be specified using the norm_value parameter.

The forward method defines the forward pass of the model. The  get_norm_layer  method is used to select the appropriate normalization layer based on the given  norm_value .

The  model_summary  function can be used to print a summary of the model architecture and the size of the input.

## Code 2: Helper Functions and Data Loading

__File:  dataLoader.py__ 

This file contains various helper functions for data loading, visualization, and augmentation.

The  load_data  function loads the CIFAR-10 dataset and applies data transformations using the Albumentations library for image augmentation. It returns the train and test data loaders.

The  plotings  class provides a method for plotting images from the dataset.

## Code 3: Training and Evaluation

__File:  performance.py__ 

This file contains the  Performance  class, which is responsible for training and evaluating the model.

The  train  method performs the training loop, iterating over the train data loader, calculating the loss, and updating the model parameters using backpropagation. It also supports L1 regularization.

The  test  method evaluates the trained model on the test data and calculates the accuracy and loss.

The  scores  function returns the training and testing loss and accuracy values.

__File: All_normalization.ipynb__

This code implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The code consists of several modules for data loading, model definition, performance evaluation, and utility functions.

The main components of the code are as follows:

__dataLoader:__ This module handles the loading and preprocessing of the CIFAR-10 dataset. It provides functions to create train and test data loaders.

__model:__ This module defines the architecture of the CNN model. It includes a Net class that represents the network structure with convolutional and normalization layers.

__performance:__ This module contains the Performance class, which is responsible for training and testing the model. It includes functions to calculate loss, accuracy, and perform model optimization.

__utils:__ This module includes utility functions used in the code, such as allocating device (GPU or CPU), model summary, and other helper functions.


