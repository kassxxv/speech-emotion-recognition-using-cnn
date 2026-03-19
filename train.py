import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import train_loader, val_loader
from models import EmotionCNNAttention
from feature_extraction import compile_features

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Extract features if not already done
if "features" not in os.listdir() or len(os.listdir("features/mel")) < 7000:
    compile_features("crema_metadata.csv")
else:
    print("Features already extracted.")


# Mixup augmentation
def mixup_data(x, y, alpha=0.4):
    """Apply mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Model with attention
model = EmotionCNNAttention(in_channels=1, num_classes=6).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss with label smoothing
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.2, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# AdamW optimizer (better for attention models)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Cosine annealing scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Training parameters
epochs = 100
best_val_loss = float("inf")
patience = 25
patience_counter = 0
mixup_alpha = 0.4

print(f"\nTraining with ATTENTION model, Mixup, label_smoothing...")

# Training loop
for epoch in range(epochs):

    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        # Apply mixup
        mixed_x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)

        optimizer.zero_grad()
        outputs = model(mixed_x)

        # Mixup loss
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation (no mixup)
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)

            loss = criterion(outputs, y)
            val_loss += loss.item()

            # Accuracy
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100.0 * correct / total

    scheduler.step()

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
        print(f"Best model saved! (val_loss: {val_loss:.4f}, acc: {val_acc:.1f}%)")
    else:
        patience_counter += 1

    print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.1f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
