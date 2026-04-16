import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from dataloader import get_loaders
from models import EmotionCNN
from feature_extraction import compile_features
from visualisation import TrainingTracker

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
parser.add_argument(
    "--normalize", action="store_true",
    help="Apply z-score normalization per sample (removes loudness bias)"
)
parser.add_argument(
    "--n-mels", type=int, default=40,
    help="Number of mel frequency bands (default: 40, try 64 for finer resolution)"
)
parser.add_argument(
    "--lr", type=float, default=0.001,
    help="Learning rate for AdamW optimizer (default: 0.001, try 0.0003)"
)
parser.add_argument(
    "--pools", type=int, choices=[3, 4], default=4,
    help="Number of MaxPool layers in the CNN (default: 4, try 3 for more spatial info)"
)
parser.add_argument(
    "--dataset", choices=["crema", "ravdess"], default="crema",
    help="Dataset to train on: 'crema' (CREMA-D) or 'ravdess' (default: crema)"
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed for reproducibility (default: 42)"
)
parser.add_argument(
    "--pretrain-from", type=str, default=None,
    help="Path to pretrained model weights to load before training (e.g. models/mel_best_model.pt)"
)
parser.add_argument(
    "--freeze-conv", action="store_true",
    help="Freeze convolutional layers, only train the FC head (use with --pretrain-from)"
)
args = parser.parse_args()

SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

feature_type = args.feature
use_dropout  = not args.no_dropout
augment_prob = 0.0 if args.no_augment else 0.7
normalize    = args.normalize
n_mels       = args.n_mels
lr           = args.lr
n_pools      = args.pools
dataset      = args.dataset

# Experiment name used for saving model and results
dataset_tag = "ravdess_"        if dataset == "ravdess"   else ""
aug_tag     = "_noaug"          if args.no_augment        else ""
dropout_tag = "_nodropout"      if args.no_dropout        else ""
norm_tag    = "_norm"           if args.normalize         else ""
mels_tag    = f"_mels{n_mels}"  if n_mels != 40           else ""
lr_tag      = f"_lr{str(lr).replace('0.', '').replace('.', '')}" if lr != 0.001 else ""
pool_tag     = f"_{n_pools}pool"  if n_pools != 4            else ""
pretrain_tag = "_transfer"        if args.pretrain_from      else ""
freeze_tag   = "_frozen"          if args.freeze_conv        else ""
exp_name    = f"{dataset_tag}{feature_type}{aug_tag}{dropout_tag}{norm_tag}{mels_tag}{lr_tag}{pool_tag}{pretrain_tag}{freeze_tag}"

# Create output directories automatically
results_dir = f"results/{exp_name}"
models_dir  = "models"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir,  exist_ok=True)

# Save seed so you always know which seed produced which result
with open(os.path.join(results_dir, "seed.txt"), "w") as f:
    f.write(str(SEED))

model_path = os.path.join(models_dir, f"{exp_name}_best_model.pt")

print(f"\nExperiment : {exp_name}")
print(f"Dataset    : {dataset}")
print(f"Seed       : {SEED}")
print(f"Feature    : {feature_type} (n_mels={n_mels})")
print(f"Dropout    : {use_dropout}")
print(f"SpecAugment: {augment_prob > 0} (prob={augment_prob})")
print(f"Normalize  : {normalize}")
print(f"LR         : {lr}")
print(f"Pools      : {n_pools}")
print(f"Model will be saved to: {model_path}")
print(f"Results will be saved to: {results_dir}/\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}") # Will use GPU if available, otherwise CPU

# Extract features if not already done
if dataset == "ravdess":
    feature_folder = f"features/ravdess/mel{n_mels}" if (feature_type == "mel" and n_mels != 40) else f"features/ravdess/{feature_type}"
    min_count = 900   # ~1080 RAVDESS speech files after 6-emotion filter
    meta_csv  = "ravdess_metadata.csv"
    feat_root = "features/ravdess"
else:
    feature_folder = f"features/mel{n_mels}" if (feature_type == "mel" and n_mels != 40) else f"features/{feature_type}"
    min_count = 7000  # CREMA-D has ~7442 files
    meta_csv  = "crema_metadata.csv"
    feat_root = "features"

if not os.path.exists(feature_folder) or len(os.listdir(feature_folder)) < min_count:
    compile_features(meta_csv, n_mels=n_mels, feature_root=feat_root)
else:
    print("Features already extracted.")

train_loader, val_loader = get_loaders(
    feature_type=feature_type,
    augment_prob=augment_prob,
    normalize=normalize,
    n_mels=n_mels,
    dataset=dataset,
)


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


# Model
model = EmotionCNN(in_channels=1, num_classes=6, use_dropout=use_dropout, n_pools=n_pools).to(device)

if args.pretrain_from:
    checkpoint   = torch.load(args.pretrain_from, map_location=device)
    model_state  = model.state_dict()
    compatible   = {k: v for k, v in checkpoint.items()
                    if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)}/{len(model_state)} layers from {args.pretrain_from}")
    if args.freeze_conv:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Frozen conv layers. Trainable params: {trainable:,}")

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss with label smoothing
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.2, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# AdamW optimizer (better for attention models)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

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
