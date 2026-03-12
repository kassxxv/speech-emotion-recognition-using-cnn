import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import train_loader, val_loader
from models import EmotionCNN
from feature_extraction import compile_features

device = "cuda" if torch.cuda.is_available() else "cpu" 
print(device)

# Extract features if not already done to prevent cpu overload during training
compile_features("ravdess_metadata.csv") if not "features" in os.listdir() else print("Features already extracted.")

# Model initialization
model = EmotionCNN().to(device)


# Loss function
criterion = nn.CrossEntropyLoss()


# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5) # weit_decay is L2 regularization to prevent overfitting


# Training parameters
epochs = 50 
best_val_loss = float("inf")


# Training loop
for epoch in range(epochs):

    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
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
            x = x.to(device)
            y = y.to(device)
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