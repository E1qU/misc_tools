# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere Univerity
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

"""

import keras.backend as K
import numpy as np
from tensorflow.keras.layers import Softmax, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical




def UARLoss(targets, inputs, eps=1e-12):
    """
    A custom UAR (unweighted average recall, also known as unweighted accuracy) loss function
    for TensorFlow. UAR is defined as the mean of class-specific recalls.
    
    Since binary values are not differentiable, the UAR computation in the loss function has
    been modified to be differentiable in the following way: in a binary case, if the ground
    truth would be 1 (i.e., the target value is [0, 1]), and the model prediction after softmax
    would be [0.8, 0.2], we interpret the prediction as 0.2 true positive and 0.8 false negative.
    Also, we want to maximize the UAR, so we want to minimize the term 1 - UAR.
    
    _____________________________________________________________________________________________
    Input parameters:
        
        
    inputs: The predicted output of the model. The format of the output should be of size
            (batch_size x num_classes). For example, in binary classification cases with a
            batch size of 1, the input could be e.g. [9.4294, 0.0313]. Please note that the loss
            function has the softmax function included (if you want to include softmax in your
            neural network, simply remove the two lines with softmax involved).
            
    targets: The ground truth values in one-hot encoding format. The format of the output
             should be of size (batch_size x num_classes). For example, in binary classification
             cases with a batch size of 1, the target could be e.g. [1, 0].
             
    eps: A small value to avoid dividing by zero when computing the recall value
    _____________________________________________________________________________________________
    
    """
    
    smax = Softmax(axis=1)
    inputs = smax(inputs)
    
    tp = K.sum(K.cast(targets * inputs, 'float'), axis=0)
    fn = K.sum(K.cast(targets * (1 - inputs), 'float'), axis=0)
    recall = tp / (tp + fn + eps)
    uar = K.mean(recall)
    
    return 1 - uar



if __name__ == '__main__':
    """
    Test out the UAR loss function with randomly generated data. If the loss value
    does not behave well, might be due to the random initialization of weights and/or
    data. In this case, just try to run the code again.
    
    """
    
    # Define the hyperparameters
    epochs = 50
    batch_size = 6
    
    # The randomly generated training samples are of size 15x25x3
    time_dim_features = 15  # The number of frames in each sample
    feature_dim = 25  # The number of features
    num_channels = 3  # The number of channels
    nb_examples_training = 120 # The number of training samples
    
    # Create our training dataset
    X_training = np.random.rand(nb_examples_training, time_dim_features, feature_dim, num_channels)
    y_training = np.random.randint(0, high=2, size=nb_examples_training)
    y_training = to_categorical(y_training, num_classes=2)
    
    # Build the model
    model = models.Sequential()
    model.add(Conv2D(24, (3, 3), activation='relu', input_shape=(15, 25, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((5, 5)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 5)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='relu'))
    
    # Compile and train the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=UARLoss)
    model.fit(X_training, y_training, epochs=epochs, batch_size=batch_size)
    
    
