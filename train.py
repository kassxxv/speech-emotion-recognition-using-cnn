import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from dataloader import get_loaders
from models import EmotionCNNAttention
from feature_extraction import compile_features
from visualisation import TrainingTracker

print(torch.__version__)
print(torch.version.cuda)

parser = argparse.ArgumentParser(description="Train SER CNN model")
parser.add_argument(
    "--feature", choices=["mel", "mfcc"], default="mel",
    help="Input feature type: 'mel' (mel-spectrogram) or 'mfcc' (default: mel)"
)
parser.add_argument(
    "--no-augment", action="store_true",
    help="Disable SpecAugment during training (ablation study)"
)
parser.add_argument(
    "--no-dropout", action="store_true",
    help="Disable dropout regularization in the model (ablation study)"
)
args = parser.parse_args()

feature_type = args.feature
use_dropout  = not args.no_dropout
augment_prob = 0.0 if args.no_augment else 0.7

# Experiment name used for saving model and results
aug_tag     = "_noaug" if args.no_augment else ""
dropout_tag = "_nodropout" if args.no_dropout else ""
exp_name    = f"{feature_type}{aug_tag}{dropout_tag}"

# Create output directories automatically
results_dir = f"results/{exp_name}"
models_dir  = "models"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir,  exist_ok=True)

model_path = os.path.join(models_dir, f"{exp_name}_best_model.pt")

print(f"\nExperiment : {exp_name}")
print(f"Feature    : {feature_type}")
print(f"Dropout    : {use_dropout}")
print(f"SpecAugment: {augment_prob > 0} (prob={augment_prob})")
print(f"Model will be saved to: {model_path}")
print(f"Results will be saved to: {results_dir}/\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}") # Will use GPU if available, otherwise CPU

# Extract features if not already done
if "features" not in os.listdir() or len(os.listdir(f"features/{feature_type}")) < 7000:
    compile_features("crema_metadata.csv")
else:
    print("Features already extracted.")

train_loader, val_loader = get_loaders(feature_type=feature_type, augment_prob=augment_prob)


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
model = EmotionCNNAttention(in_channels=1, num_classes=6, use_dropout=use_dropout).to(device) # added to model new flag use_dropout
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
best_val_macro_f1 = 0.0 # Track best macro-F1 for balanced emotion performance
patience = 25 # Early stopping parameter (stop if no improvement in macro-F1 for 25 epochs)
patience_counter = 0 # Counter for early stopping
mixup_alpha = 0.4

tracker = TrainingTracker(name=exp_name, output_dir=results_dir)

print(f"\nTraining with feature={feature_type}, dropout={use_dropout}, augment_prob={augment_prob}...")

# Training loop
for epoch in range(epochs):

    model.train() # Set model to training mode for dropout and batchnorm
    train_loss = 0 # Track training loss for monitoring

    for x, y in train_loader:
        x = x.to(device)  # Move input to device (GPU/CPU)
        y = y.to(device) #  Move labels to device (GPU/CPU)

        # Apply mixup
        mixed_x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)

        optimizer.zero_grad() # Zero gradients before backpropagation
        outputs = model(mixed_x) # Forward pass with mixed inputs

        # Mixup loss
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward() # Backpropagation to compute gradients
        optimizer.step() # Update model parameters

        train_loss += loss.item() # Accumulate training loss for monitoring

    train_loss /= len(train_loader)

    # Validation (no mixup)
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad(): # Disable gradient computation for validation (saves memory and computations)
        for x, y in val_loader:
            x = x.to(device) # Move input to device (GPU/CPU)
            y = y.to(device) # Move labels to device (GPU/CPU)
            outputs = model(x) # Forward pass on validation data

            loss = criterion(outputs, y) # Compute validation loss (no mixup, so standard criterion)
            val_loss += loss.item() # Accumulate validation loss for monitoring

            # Accuracy
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = 100.0 * correct / total
    val_f1_macro = f1_score(y_true, y_pred, average="macro")
    scheduler.step()

    # Save best model by macro-F1 to target balanced emotion performance.
    if val_f1_macro > best_val_macro_f1:
        best_val_macro_f1 = val_f1_macro
        patience_counter = 0
        torch.save(model.state_dict(), model_path) # Save best model weights
        print(
            f"Best model saved! "
            f"(macro_f1: {val_f1_macro:.4f}, acc: {val_acc:.1f}%)"
        )
    else:
        patience_counter += 1
    tracker.log(train_loss, val_loss, val_f1_macro)
    print(
        f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | ValLoss: {val_loss:.4f} "
        f"| Acc: {val_acc:.1f}% | F1(m): {val_f1_macro:.4f} "
        f"| LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\nTraining complete. Best macro-F1: {best_val_macro_f1:.4f}")
tracker.plot()
tracker.plot_f1()
