import os
import argparse
import torch
import numpy as np
import pandas as pd
import librosa
from models import EmotionCNN
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import load_compatible_state_dict, add_gaussian_noise, extract_feature_from_waveform

parser = argparse.ArgumentParser(description="Evaluate SER model: noise robustness + confusion matrix")
parser.add_argument(
    "--feature", choices=["mel", "mfcc"], default="mel",
    help="Feature type used during training (default: mel)"
)
parser.add_argument(
    "--no-augment", action="store_true",
    help="Evaluate the model trained without SpecAugment (ablation study)"
)
parser.add_argument(
    "--no-dropout", action="store_true",
    help="Load a model trained without dropout (must match training flag used)"
)
parser.add_argument(
    "--dataset", choices=["crema", "ravdess"], default="crema",
    help="Dataset the model was trained on (default: crema)"
)
parser.add_argument(
    "--normalize", action="store_true",
    help="Load a model trained with z-score normalization"
)
parser.add_argument(
    "--n-mels", type=int, default=40,
    help="Number of mel bands used during training (default: 40)"
)
parser.add_argument(
    "--lr", type=float, default=0.001,
    help="Learning rate used during training (default: 0.001)"
)
parser.add_argument(
    "--pools", type=int, choices=[3, 4], default=4,
    help="Number of MaxPool layers used during training (default: 4)"
)
parser.add_argument(
    "--pretrain-from", type=str, default=None,
    help="Used during training — needed to reconstruct the experiment name"
)
parser.add_argument(
    "--freeze-conv", action="store_true",
    help="Used during training — needed to reconstruct the experiment name"
)
args = parser.parse_args()

feature_type  = args.feature
dataset       = args.dataset
n_mels        = args.n_mels
lr            = args.lr
n_pools       = args.pools
dataset_tag   = "ravdess_" if dataset == "ravdess" else ""
aug_tag       = "_noaug"    if args.no_augment      else ""
dropout_tag   = "_nodropout" if args.no_dropout     else ""
norm_tag      = "_norm"     if args.normalize       else ""
mels_tag      = f"_mels{n_mels}" if n_mels != 40   else ""
lr_tag        = f"_lr{str(lr).replace('0.', '').replace('.', '')}" if lr != 0.001 else ""
pool_tag      = f"_{n_pools}pool" if n_pools != 4   else ""
pretrain_tag  = "_transfer" if args.pretrain_from   else ""
freeze_tag    = "_frozen"   if args.freeze_conv     else ""
exp_name      = f"{dataset_tag}{feature_type}{aug_tag}{dropout_tag}{norm_tag}{mels_tag}{lr_tag}{pool_tag}{pretrain_tag}{freeze_tag}"

results_dir  = f"results/{exp_name}"
model_path   = f"models/{exp_name}_best_model.pt"
os.makedirs(results_dir, exist_ok=True)

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Loading model: {model_path}")

# Load model
model = EmotionCNN(in_channels=1, num_classes=6, n_pools=n_pools).to(device)


try:
    skipped_keys = load_compatible_state_dict(model, model_path, device)
    print("Model loaded successfully (compatible weights).")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} incompatible/missing keys (example: {skipped_keys[:4]})")
except FileNotFoundError:
    print(f"ERROR: {model_path} not found. Run train.py --feature {feature_type} first.")
    exit()

model.eval()


def get_test_dataframe(csv_path="crema_metadata.csv", split_seed=42):
    """Replicate dataloader test actor split for held-out evaluation."""
    df = pd.read_csv(csv_path)
    all_actors = sorted(df["actor_id"].unique())
    n_actors = len(all_actors)
    rng = np.random.default_rng(split_seed)
    shuffled_actors = rng.permutation(all_actors)
    n_train = int(n_actors * 0.70)
    n_val   = int(n_actors * 0.15)
    test_actors = shuffled_actors[n_train + n_val:]  # same 15% held-out split as dataloader
    return df[df["actor_id"].isin(test_actors)].reset_index(drop=True)


# Evaluate clean + noisy conditions
evaluation_conditions = [("clean", None), ("snr_20", 20), ("snr_5", 5)]
f1_scores = []
condition_labels = []
csv_path = "ravdess_metadata.csv" if dataset == "ravdess" else "crema_metadata.csv"
val_df   = get_test_dataframe(csv_path=csv_path)

print(f"Test samples for evaluation: {len(val_df)}")

# Store clean predictions for confusion matrix
clean_y_true = []
clean_y_pred = []

np.random.seed(42)
for label, snr in evaluation_conditions:
    if snr is None:
        print("\nEvaluating clean input")
    else:
        print(f"\nEvaluating with SNR = {snr} dB")

    y_true = []
    y_pred = []

    with torch.no_grad():
        for row in val_df.itertuples(index=False):
            y_wave, sr = librosa.load(row.file_path, sr=None)
            if snr is not None:
                y_wave = add_gaussian_noise(y_wave, snr)

            feature = extract_feature_from_waveform(y_wave, sr, feature_type)
            x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            outputs = model(x)
            pred = int(outputs.argmax(dim=1).item())

            y_true.append(int(row.emotion_id))
            y_pred.append(pred)

    f1 = f1_score(y_true, y_pred, average="macro")
    f1_scores.append(f1)
    condition_labels.append(label)

    if snr is None:
        print(f"F1-score (clean): {f1:.4f}")
        clean_y_true = y_true # Save clean predictions for confusion matrix
        clean_y_pred = y_pred
    else:
        print(f"F1-score (SNR={snr}): {f1:.4f}")


# Plot noise robustness
plt.figure(figsize=(8, 5))
plt.plot(condition_labels, f1_scores, marker='o', linewidth=2, markersize=8)
plt.xlabel("Condition", fontsize=12)
plt.ylabel("F1-score", fontsize=12)
plt.title(f"Model Performance: Clean vs Noisy ({feature_type.upper()})", fontsize=14)
plt.grid(True, alpha=0.3)
noise_plot_path = os.path.join(results_dir, "noise_robustness.png")
plt.savefig(noise_plot_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"\nNoise robustness plot saved to {noise_plot_path}")

# Confusion matrix (clean condition)
cm = confusion_matrix(clean_y_true, clean_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTION_NAMES)
fig, ax = plt.subplots(figsize=(8, 7))
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Confusion Matrix — {feature_type.upper()} (clean)", fontsize=14)
plt.tight_layout()
cm_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"Confusion matrix saved to {cm_path}")

# Per-class F1 for error analysis
per_class_f1 = f1_score(clean_y_true, clean_y_pred, average=None)
print("\nPer-class F1 (clean):")
for name, score in zip(EMOTION_NAMES, per_class_f1):
    print(f"  {name:<10} {score:.4f}")

# Save per-class F1 to CSV
per_class_df = pd.DataFrame({
    "emotion":  EMOTION_NAMES,
    "f1_score": per_class_f1.round(4),
})
per_class_path = os.path.join(results_dir, "per_class_f1.csv")
per_class_df.to_csv(per_class_path, index=False)
print(f"Per-class F1 saved to {per_class_path}")

# Append this run to the global comparison table
comparison_path = "results/comparison_table.csv"
clean_f1_weighted = f1_scores[0]
snr20_f1          = f1_scores[1]
snr5_f1           = f1_scores[2]

new_row = pd.DataFrame([{
    "experiment":   exp_name,
    "dataset":      dataset,
    "feature":      feature_type,
    "augment":      not args.no_augment,
    "dropout":      not args.no_dropout,
    "clean_f1":     round(clean_f1_weighted, 4),
    "snr20_f1":     round(snr20_f1, 4),
    "snr5_f1":      round(snr5_f1, 4),
}])

if os.path.exists(comparison_path):
    existing = pd.read_csv(comparison_path)
    # Replace row if experiment already exists, otherwise append
    existing = existing[existing["experiment"] != exp_name]
    comparison_df = pd.concat([existing, new_row], ignore_index=True)
else:
    comparison_df = new_row

comparison_df.to_csv(comparison_path, index=False)
print(f"\nComparison table updated: {comparison_path}")
print(comparison_df.to_string(index=False))
