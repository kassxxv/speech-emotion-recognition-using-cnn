import torch
import numpy as np
import pandas as pd
import librosa
from models import EmotionCNNAttention
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Device
device = "Cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load attention model
model = EmotionCNNAttention(in_channels=1, num_classes=6).to(device)


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


try:
    skipped_keys = load_compatible_state_dict(model, "best_model.pt", device)
    print("Model loaded successfully (compatible weights).")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} incompatible/missing keys (example: {skipped_keys[:4]})")
except FileNotFoundError:
    print("ERROR: best_model.pt not found.")
    exit()

model.eval()


def add_noise_snr(signal, snr_db):
    """Add Gaussian noise with specified SNR (Signal-to-Noise Ratio) in dB."""
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def extract_mel_from_waveform(y, sr, n_mels=40, target_frames=200):
    """Create a fixed-size log-mel spectrogram from waveform."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)

    if mel_db.shape[1] < target_frames:
        pad = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)))
    else:
        mel_db = mel_db[:, :target_frames]

    return mel_db.astype(np.float32)


def get_val_dataframe(csv_path="crema_metadata.csv", split_seed=42):
    """Replicate dataloader validation actor split for consistent evaluation."""
    df = pd.read_csv(csv_path)
    all_actors = sorted(df["actor_id"].unique())
    rng = np.random.default_rng(split_seed)
    shuffled_actors = rng.permutation(all_actors)
    val_actors = shuffled_actors[int(len(all_actors) * 0.8):]
    return df[df["actor_id"].isin(val_actors)].reset_index(drop=True)


# Evaluate clean + noisy conditions
evaluation_conditions = [("clean", None), ("snr_20", 20), ("snr_5", 5)]
f1_scores = []
condition_labels = []
val_df = get_val_dataframe()

print(f"Validation samples for evaluation: {len(val_df)}")

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
                y_wave = add_noise_snr(y_wave, snr)

            mel = extract_mel_from_waveform(y_wave, sr)
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            outputs = model(x)
            pred = int(outputs.argmax(dim=1).item())

            y_true.append(int(row.emotion_id))
            y_pred.append(pred)

    f1 = f1_score(y_true, y_pred, average="weighted")
    f1_scores.append(f1)
    condition_labels.append(label)

    if snr is None:
        print(f"F1-score (clean): {f1:.4f}")
    else:
        print(f"F1-score (SNR={snr}): {f1:.4f}")


# Plot results
plt.figure(figsize=(8, 5))
plt.plot(condition_labels, f1_scores, marker='o', linewidth=2, markersize=8)
plt.xlabel("Condition", fontsize=12)
plt.ylabel("F1-score", fontsize=12)
plt.title("Model Performance: Clean vs Noisy", fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig("noise_robustness.png", dpi=100, bbox_inches='tight')
print("\nPlot saved to noise_robustness.png")
