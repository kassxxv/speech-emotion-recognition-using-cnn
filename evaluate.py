import torch
import numpy as np
from models import EmotionCNN
from dataloader import val_loader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from audiomentations import AddGaussianNoise

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model = EmotionCNN().to(device)

try:
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("ERROR: best_model.pt not found.")
    exit()

model.eval()


# --- Функция добавления шума с контролем SNR ---
def add_noise_snr(signal, snr_db):
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


# --- Оценка при разных SNR ---
snr_levels = [20, 5]  # dB
f1_scores = []

for snr in snr_levels:
    print(f"\nEvaluating with SNR = {snr} dB")

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            # Добавляем шум к входу
            noisy_x = x.cpu().numpy()
            noisy_x = np.array([add_noise_snr(sample, snr) for sample in noisy_x])
            noisy_x = torch.tensor(noisy_x, dtype=torch.float32).to(device)

            outputs = model(noisy_x)
            preds = outputs.argmax(dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    f1 = f1_score(y_true, y_pred, average="weighted")
    f1_scores.append(f1)

    print(f"F1-score (SNR={snr}): {f1:.4f}")


# --- График ---
plt.figure()
plt.plot(snr_levels, f1_scores, marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("F1-score")
plt.title("Model Robustness to Noise")
plt.gca().invert_xaxis()  # чтобы 20 → 5 шло слева направо
plt.grid()
plt.savefig("noise_robustness.png")
plt.show()