import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# pytorch-grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import librosa

from models import EmotionCNNAttention
from feature_extraction import extract_mel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CHECKPOINT_PATH = "best_model.pt"
CSV_PATH        = "crema_metadata.csv"
TARGET_EMOTION  = "ANG"          # Emotion code from CREMA-D (ANG/DIS/FEA/HAP/NEU/SAD)
OUTPUT_PATH     = "gradcam_anger.png"

EMOTION_MAP = {
    "ANG": ("angry",   0),
    "DIS": ("disgust", 1),
    "FEA": ("fear",    2),
    "HAP": ("happy",   3),
    "NEU": ("neutral", 4),
    "SAD": ("sad",     5),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# 1. LOAD MODEL
#    Reusing load_compatible_state_dict from evaluate.py
# ─────────────────────────────────────────────
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
    skipped = load_compatible_state_dict(model, CHECKPOINT_PATH, device)
    print(f"Model loaded. Skipped keys: {len(skipped)}")
except FileNotFoundError:
    print(f"ERROR: {CHECKPOINT_PATH} not found. Run train.py first.")
    exit()

model.eval()


# ─────────────────────────────────────────────
# 2. FIND A SINGLE AUDIO FILE FOR TARGET EMOTION
#    Using crema_metadata.csv to locate the file
# ─────────────────────────────────────────────
def find_file_for_emotion(csv_path, emotion_code):
    """Find the first audio file in the CSV matching the given emotion code."""
    df = pd.read_csv(csv_path)
    row = df[df["emotion_code"] == emotion_code].iloc[0]
    return row["file_path"], row["emotion_id"], row["emotion_name"]


wav_path, true_label, emotion_name = find_file_for_emotion(CSV_PATH, TARGET_EMOTION)
print(f"File: {wav_path}")
print(f"Emotion: {emotion_name} (class {true_label})")


# ─────────────────────────────────────────────
# 3. EXTRACT MEL SPECTROGRAM
#    Reusing extract_mel() from feature_extraction.py
#    Returns np.array of shape (40, 200) — (frequencies, time)
# ─────────────────────────────────────────────
mel_np = extract_mel(wav_path)          # shape: (40, 200), dtype float32

# Normalize to [0, 1] for RGB overlay visualization
mel_min, mel_max = mel_np.min(), mel_np.max()
mel_norm = (mel_np - mel_min) / (mel_max - mel_min + 1e-8)  # (40, 200) in [0, 1]

# Add batch and channel dimensions for the model: (1, 1, 40, 200)
input_tensor = torch.tensor(mel_np).float().unsqueeze(0).unsqueeze(0).to(device)


# ─────────────────────────────────────────────
# 4. GRAD-CAM
#    target_layer = last Conv2d = model.conv4
#    ClassifierOutputTarget(class_idx) — explain the chosen class
# ─────────────────────────────────────────────
target_layer = model.conv4   # last Conv2d layer (128 -> 256 channels)

# Get model prediction before computing Grad-CAM
with torch.no_grad():
    logits = model(input_tensor)
    pred_class = int(logits.argmax(dim=1).item())
    pred_name  = [v[0] for v in EMOTION_MAP.values() if v[1] == pred_class][0]

print(f"Predicted class: {pred_name} ({pred_class}), True: {emotion_name} ({true_label})")

# Create GradCAM object
#   target_layers — must be a list, even for a single layer
#   hooks are registered automatically, no manual intervention needed
cam = GradCAM(
    model=model,
    target_layers=[target_layer],
)

# Use the ground-truth class as the explanation target
# Alternatively: targets=None to explain the predicted class
targets = [ClassifierOutputTarget(true_label)]

# Run Grad-CAM: returns np.array of shape (1, H, W) -> take [0] -> (H, W)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0]   # shape: (40, 200), values in [0, 1]

print(f"Grad-CAM map shape: {grayscale_cam.shape}")
print(f"Grad-CAM min={grayscale_cam.min():.3f}, max={grayscale_cam.max():.3f}")


# ─────────────────────────────────────────────
# 5. VISUALIZATION AND SAVE PNG
#    Convert mel_norm to 3-channel RGB for overlay
# ─────────────────────────────────────────────

# Apply viridis colormap to mel spectrogram to get an RGB image
mel_rgb = cm.viridis(mel_norm)[:, :, :3]   # (40, 200, 3), float64 in [0, 1]
mel_rgb = mel_rgb.astype(np.float32)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1 — Original mel spectrogram
axes[0].imshow(mel_np, aspect="auto", origin="lower", cmap="viridis")
axes[0].set_title("Mel Spectrogram (original)", fontsize=13)
axes[0].set_xlabel("Time (frames)")
axes[0].set_ylabel("Mel frequency (bin)")
plt.colorbar(axes[0].images[0], ax=axes[0], label="dB")

# Panel 2 — Raw Grad-CAM heatmap (grayscale rendered as jet colormap)
im = axes[1].imshow(grayscale_cam, aspect="auto", origin="lower", cmap="jet")
axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
axes[1].set_xlabel("Time (frames)")
axes[1].set_ylabel("Mel frequency (bin)")
plt.colorbar(im, ax=axes[1], label="Importance")

# Panel 3 — Overlay: spectrogram + heatmap blended together
axes[2].imshow(mel_rgb, aspect="auto", origin="lower")
axes[2].imshow(grayscale_cam, aspect="auto", origin="lower",
               cmap="jet", alpha=0.5)   # alpha=0.5 blends heatmap over spectrogram
axes[2].set_title(
    f"Overlay\nTrue: {emotion_name} | Predicted: {pred_name}",
    fontsize=12
)
axes[2].set_xlabel("Time (frames)")
axes[2].set_ylabel("Mel frequency (bin)")

plt.suptitle(
    f"Grad-CAM — emotion '{emotion_name}' | conv4 (last Conv2d layer)",
    fontsize=14, fontweight="bold", y=1.02
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_PATH}")
plt.show()
