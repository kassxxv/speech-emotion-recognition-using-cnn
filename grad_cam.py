import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import soundfile as sf

# pytorch-grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import librosa

from models import EmotionCNNAttention
from feature_extraction import extract_mel, extract_mfcc

parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for all emotions")
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
args = parser.parse_args()

feature_type = args.feature
aug_tag      = "_noaug" if args.no_augment else ""
dropout_tag  = "_nodropout" if args.no_dropout else ""
exp_name     = f"{feature_type}{aug_tag}{dropout_tag}"

model_path  = f"models/{exp_name}_best_model.pt"
output_dir  = f"results/{exp_name}/gradcam"
os.makedirs(output_dir, exist_ok=True)

EMOTION_MAP = {
    "ANG": ("angry",   0),
    "DIS": ("disgust", 1),
    "FEA": ("fear",    2),
    "HAP": ("happy",   3),
    "NEU": ("neutral", 4),
    "SAD": ("sad",     5),
}

CSV_PATH   = "crema_metadata.csv"
HOP_LENGTH = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Loading model: {model_path}")


def load_compatible_state_dict(model, checkpoint_path, device):
    """Load only checkpoint parameters that exist in the current model with matching shape."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    model.load_state_dict(compatible, strict=False)
    return skipped


model = EmotionCNNAttention(in_channels=1, num_classes=6).to(device)

try:
    skipped = load_compatible_state_dict(model, model_path, device)
    print(f"Model loaded. Skipped keys: {len(skipped)}")
except FileNotFoundError:
    print(f"ERROR: {model_path} not found. Run train.py --feature {feature_type} first.")
    exit()

model.eval()


def find_file_for_emotion(csv_path, emotion_code):
    """Find the first audio file in the CSV matching the given emotion code."""
    df = pd.read_csv(csv_path)
    row = df[df["emotion_code"] == emotion_code].iloc[0]
    return row["file_path"], row["emotion_id"], row["emotion_name"]


def extract_feature(wav_path, feature_type):
    """Extract mel or MFCC feature array from wav file."""
    if feature_type == "mel":
        return extract_mel(wav_path)   # shape: (40, 200)
    else:
        return extract_mfcc(wav_path)  # shape: (40, 200)


def run_gradcam(emotion_code):
    """Generate and save a Grad-CAM visualization for one emotion."""
    emotion_name, true_label = EMOTION_MAP[emotion_code]

    wav_path, _, _ = find_file_for_emotion(CSV_PATH, emotion_code)
    print(f"\n[{emotion_code}] File: {wav_path}")

    # Extract feature
    feature_np = extract_feature(wav_path, feature_type)  # (40, 200)

    # Normalize to [0, 1] for RGB overlay visualization
    feat_min, feat_max = feature_np.min(), feature_np.max()
    feat_norm = (feature_np - feat_min) / (feat_max - feat_min + 1e-8)

    # Add batch and channel dimensions for the model: (1, 1, 40, 200)
    input_tensor = torch.tensor(feature_np).float().unsqueeze(0).unsqueeze(0).to(device)

    # Detect actual audio duration for correct time axis labeling
    info       = sf.info(wav_path)
    real_sec   = info.duration
    SR         = info.samplerate
    last_frame = min(int(real_sec * SR / HOP_LENGTH), feature_np.shape[1])
    real_dur   = last_frame * HOP_LENGTH / SR

    # Model prediction
    with torch.no_grad():
        logits = model(input_tensor)
        pred_class = int(logits.argmax(dim=1).item())
        pred_name  = [v[0] for v in EMOTION_MAP.values() if v[1] == pred_class][0]

    print(f"  Predicted: {pred_name} ({pred_class}), True: {emotion_name} ({true_label})")

    # Grad-CAM — target layer: conv4 (last Conv2d, 128→256 channels)
    target_layer = model.conv4
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(true_label)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]  # shape: (40, 200), values in [0, 1]

    # Crop to actual audio content (remove padding tail)
    feat_crop = feature_np[:, :last_frame]
    cam_crop  = grayscale_cam[:, :last_frame]
    rgb_crop  = cm.viridis(feat_norm[:, :last_frame])[:, :, :3].astype(np.float32)

    # Time tick labels
    n_cols   = feat_crop.shape[1]
    tick_pos = np.linspace(0, n_cols - 1, 6)
    tick_sec = [f"{t:.2f}" for t in tick_pos * HOP_LENGTH / SR]

    y_label = "Mel frequency (bin)" if feature_type == "mel" else "MFCC coefficient"

    # 3-panel figure: original | Grad-CAM | overlay
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 — Original feature
    img1 = axes[0].imshow(feat_crop, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_xticks(tick_pos)
    axes[0].set_xticklabels(tick_sec)
    axes[0].set_title(f"{feature_type.upper()} (original)", fontsize=13)
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel(y_label)
    plt.colorbar(img1, ax=axes[0], label="dB" if feature_type == "mel" else "Value")

    # Panel 2 — Grad-CAM heatmap
    img2 = axes[1].imshow(cam_crop, aspect="auto", origin="lower", cmap="jet")
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels(tick_sec)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel(y_label)
    plt.colorbar(img2, ax=axes[1], label="Importance")

    # Panel 3 — Overlay
    axes[2].imshow(rgb_crop, aspect="auto", origin="lower")
    axes[2].imshow(cam_crop, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[2].set_xticks(tick_pos)
    axes[2].set_xticklabels(tick_sec)
    axes[2].set_title(f"Overlay — True: {emotion_name} | Predicted: {pred_name}", fontsize=13)
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel(y_label)

    plt.suptitle(
        f"Grad-CAM — '{emotion_name}' | conv4 (last Conv2d layer) | {feature_type.upper()}",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"gradcam_{emotion_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# Run Grad-CAM for all 6 emotions
for emotion_code in EMOTION_MAP:
    run_gradcam(emotion_code)

print("\nDone.")
