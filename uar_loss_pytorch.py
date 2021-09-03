# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere Univerity
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

"""


import torch
import numpy as np
from torch import cuda, rand, randint, Tensor, flatten
from torch.nn import Module, Softmax, Linear, ReLU, Conv2d, BatchNorm2d, MaxPool2d
from torch.nn.functional import one_hot
from torch.optim import Adam



class UARLoss(Module):
    def __init__(self):
        super(UARLoss, self).__init__()

    def forward(self, inputs, targets, eps=1e-12):
        """
        A custom UAR (unweighted average recall, also known as unweighted accuracy) loss function
        for PyTorch. UAR is defined as the mean of class-specific recalls.
        
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
                neural network, simply remove the two lines with softmax involved). In addition,
                note that "inputs" should be a Torch tensor in order to be able to use PyTorch's
                autograd functionality for backpropagation.
                
        targets: The ground truth values in one-hot encoding format. The format of the output
                 should be of size (batch_size x num_classes). For example, in binary classification
                 cases with a batch size of 1, the target could be e.g. [1, 0]. Please note that
                 "targets" should be a Torch tensor.
                 
        eps: A small value to avoid dividing by zero when computing the recall value
        _____________________________________________________________________________________________
        
        """
        
        smax = Softmax(dim=1)
        inputs = smax(inputs)
        targets = targets.to(torch.float32)
        
        tp = (targets * inputs).sum(dim=0).to(torch.float32)
        fn = (targets * (1 - inputs)).sum(dim=0).to(torch.float32)
        recall = tp / (tp + fn + eps)
        uar = recall.mean()
        
        return 1 - uar








class SimpleDNN(Module):

    def __init__(self) -> None:

        super().__init__()
        
        self.conv_layer_1 = Conv2d(in_channels=3, out_channels=24, kernel_size=3, padding=1)
        self.batch_norm_1 = BatchNorm2d(24)
        self.maxpool_1 = MaxPool2d((5,5))
        
        self.conv_layer_2 = Conv2d(in_channels=24, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm_2 = BatchNorm2d(64)
        self.maxpool_2 = MaxPool2d((3,5))
        
        self.linear_layer_1 = Linear(in_features=64, out_features=32)
        self.linear_layer_2 = Linear(in_features=32, out_features=2)
        
        self.relu = ReLU()
        
        
    def forward(self, X: Tensor) -> Tensor:
        
        X = self.maxpool_1(self.relu(self.batch_norm_1(self.conv_layer_1(X))))
        X = self.maxpool_2(self.relu(self.batch_norm_2(self.conv_layer_2(X))))
        X = flatten(X,start_dim=1,end_dim=-1)
        X = self.relu(self.linear_layer_1(X))
        X = self.relu(self.linear_layer_2(X))
        
        return X


if __name__ == '__main__':
    """
    Test out the UAR loss function with randomly generated data. If the loss value
    does not behave well, might be due to the random initialization of weights and/or
    data. In this case, just try to run the code again.
    
    """
    
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')        
    
    # Define the hyperparameters
    epochs = 50
    batch_size = 6
    
    # The randomly generated training samples are of size 3x15x25
    num_channels = 3  # The number of channels
    time_dim_features = 15  # The number of frames in each sample
    feature_dim = 25  # The number of features
    nb_examples_training = 120 # The number of training samples
    
    # Initialize our DNN
    example_dnn = SimpleDNN()
    
    # Pass DNN to the available device.
    example_dnn = example_dnn.to(device)

    # Give the parameters of our DNN to an optimizer.
    optimizer = Adam(params=example_dnn.parameters(), lr=1e-3)

    # Initialize our loss function
    loss_function = UARLoss()

    # Create our training dataset
    X_training = rand(nb_examples_training, num_channels, time_dim_features, feature_dim)
    y_training = randint(0, 2, (nb_examples_training,))
    y_training = one_hot(y_training)
    
    # Start the training process
    for epoch in range(1, epochs+1):

        # A list for the losses of each minibatch
        epoch_loss_training = []

        # Enter training mode for our data
        example_dnn.train()

        # Go through each batch of our dataset
        for i in range(0, nb_examples_training, batch_size):
            
            # Zero the gradient of the optimizer
            optimizer.zero_grad()

            # Get the data for our batch
            X_input = X_training[i:i+batch_size, :]
            y_output = y_training[i:i+batch_size]

            # Pass the data the appropriate device
            X_input = X_input.to(device)
            y_output = y_output.to(device)

            # Get the output of our model
            y_hat = example_dnn(X_input)

            # Calculate the loss of our model based on the UAR loss function
            loss = loss_function(y_hat, y_output)

            # Perform the backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Add the loss of the batch to the list of losses
            epoch_loss_training.append(loss.item())

        # Calculate the mean loss for the epoch
        epoch_loss_training = np.array(epoch_loss_training).mean()

        print(f'Epoch: {epoch:03d} | Mean training loss: {epoch_loss_training:7.4f}')        