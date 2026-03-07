import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from models import EmotionCNN


# -----------------------------
# Dummy dataset (тимчасово)
# -----------------------------
class DummyDataset(Dataset):

    def __len__(self):
        return 200

    def __getitem__(self, idx):

        x = torch.randn(1, 40, 200) # Randomly generated spectrogram (1 channel, 40 mel bands, 200 time frames)
        y = torch.randint(0, 8, (1,)).item()  # emotion label

        return x, y


# Data loaders
train_dataset = DummyDataset()
val_dataset = DummyDataset()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Model initialization
model = EmotionCNN()


# Loss function
criterion = nn.CrossEntropyLoss()


# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training parameters
epochs = 10
best_val_loss = float("inf")


# Training loop
for epoch in range(epochs):

    model.train()
    train_loss = 0

    for x, y in train_loader:

        optimizer.zero_grad() # Clears the gradients of all optimized tensors

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward() # Computes the gradient of the loss with respect to the model parameters

        optimizer.step() # Updates the model parameters based on the computed gradients

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():

        for x, y in val_loader:

            outputs = model(x)

            loss = criterion(outputs, y)

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # Save the best model based on validation loss
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(model.state_dict(), "best_model.pt")

        print("Best model saved!")


    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")