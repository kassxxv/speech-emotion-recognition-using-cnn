import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        self.dropout2d = nn.Dropout2d(0.1)  # Reduced from 0.2
        self.dropout = nn.Dropout(0.4)       # Reduced from 0.5

        # After 3 pooling: 40x200 -> 20x100 -> 10x50 -> 5x25
        self.fc1 = nn.Linear(128 * 5 * 25, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # No dropout on first layer - let it learn basic features

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x