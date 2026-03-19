import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Attention mechanism to focus on important time frames."""

    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 1, 1)
        )

    def forward(self, x):
        # x: (batch, channels, freq, time)
        weights = self.attention(x)  # (batch, 1, freq, time)
        weights = F.softmax(weights.view(x.size(0), -1), dim=1)
        weights = weights.view_as(weights.unsqueeze(1).expand_as(x[:, :1, :, :]))
        weights = weights.unsqueeze(1).expand_as(x)

        # Weighted sum
        attended = (x * weights).sum(dim=[2, 3])  # (batch, channels)
        return attended


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, channels, freq, time)
        b, c, _, _ = x.size()
        # Global average pooling
        y = x.view(b, c, -1).mean(dim=2)  # (batch, channels)
        # Channel weights
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class EmotionCNNAttention(nn.Module):
    """CNN with attention for emotion recognition."""

    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()
        # Conv block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

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
        self.dropout = nn.Dropout(0.5)

        # Attention modules
        self.channel_attention = ChannelAttention(256, reduction=16)

        # Global average pooling + FC
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)

        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2d(x)

        # Channel attention
        x = self.channel_attention(x)

        # Global pooling instead of flatten (more robust)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Keep old model for compatibility
class EmotionCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)
        self.dropout2d = nn.Dropout2d(0.15)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * 2 * 12, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2d(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
