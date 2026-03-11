import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    # Initializes the CNN architecture
    def __init__(self):
        # Parent class constructor
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # Kernel, input channels, output channels, kernel size 3x3, padding (frame of the image)
        self.bn1 = nn.BatchNorm2d(32) # Batch normalization for 32 channels

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # Kernel, input channels, output channels, kernel size 3x3, padding (frame of the image)
        self.bn2 = nn.BatchNorm2d(64) # Batch normalization for 64 channels

        # Pooling layer
        self.pool = nn.MaxPool2d(2) # Reduces the spatial dimensions by 2 (e.g., from 20x100 to 10x50)
        # Dropout layer
        self.dropout = nn.Dropout(0.5) # Turns off 50% of the neurons during training to prevent overfitting
        # Final classifier
        self.fc1 = nn.Linear(64*10*50, 128) # Fully connected layer, input size is 64 channels * 13 height * 50 width (after pooling), output size is 128
        self.fc2 = nn.Linear(128, 8) # Input size is 128, output size is 8 (number of emotion classes)

    def forward(self, x):
    # Passes the input through the first convolutional layer, applies batch normalization, ReLU activation, and then max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
    
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
    # Flattens the output from the convolutional layers to feed into the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x))) # First layer with dropout and ReLU activation
        # Output layer with 8 classes (emotions)
        x = self.fc2(x)
        return x