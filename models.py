import torch.nn as nn
import torch.nn.functional as F


class EmotionCNNAttention(nn.Module):
    """CNN for emotion recognition."""

    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()
        # Conv block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) 

        # Batch normalization after conv layer for better training stability and faster convergence

        # Conv block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Conv block 4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)
        self.dropout2d = nn.Dropout2d(0.15)
        self.dropout = nn.Dropout(0.5) # Dropout 50% neurons for regularization to prevent overfitting

        # Global average pooling + FC

        self.global_pool = nn.AdaptiveAvgPool2d(1) # Global pooling instead of flatten (more robust) 
        self.fc1 = nn.Linear(256, 256) # Hidden layer with 256 units for better representation learning
        self.fc2 = nn.Linear(256, num_classes) # Output layer with 6 classes (emotions)

    def forward(self, x):
        # Conv blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 200x40 -> 100x20

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # 100x20 -> 50x10
        x = self.dropout2d(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # 50x10 -> 25x5
        x = self.dropout2d(x)

        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # 25x5 -> 12x2
        x = self.dropout2d(x)

        # Global pooling instead of flatten (more robust)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Alternative simpler CNN without attention (for ablation study)
# class EmotionCNN(nn.Module):
#     def __init__(self, in_channels=1, num_classes=6):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)

#         self.pool = nn.MaxPool2d(2)
#         self.dropout2d = nn.Dropout2d(0.15)
#         self.dropout = nn.Dropout(0.5)

#         self.fc1 = nn.Linear(256 * 2 * 12, 512) # Hidden layer with 512 units for better representation learning
#         self.fc2 = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.dropout2d(x)
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.dropout2d(x)
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         x = self.dropout2d(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x
