import os
import numpy as np
import torch
import torch.nn.functional as F

from models import EmotionCNN
from utils import load_compatible_state_dict, extract_feature_from_waveform

_CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'model_weights', 'mel_best_model.pt')
_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

# Per-class logit bias applied before softmax.
# Negative value suppresses that class, positive boosts it.
# Order: Angry, Disgust, Fear, Happy, Neutral, Sad
_LOGIT_BIAS = torch.tensor([[-0.6, 0.0, 0.0, 0.0, 0.0, 0.0]])


class EmotionPredictor:
    def __init__(self, checkpoint: str = _CHECKPOINT):
        self.device = torch.device('cpu')
        self.model = EmotionCNN(in_channels=1, num_classes=6)
        skipped = load_compatible_state_dict(self.model, checkpoint, self.device)
        if skipped:
            print(f"[inference] skipped keys: {skipped}")
        self.model.eval()

    def predict(self, audio_np: np.ndarray, sample_rate: int) -> dict:
        feat = extract_feature_from_waveform(audio_np, sample_rate, 'mel')
        tensor = torch.tensor(feat).unsqueeze(0).unsqueeze(0)  # (1, 1, 40, 200)
        with torch.no_grad():
            logits = self.model(tensor) + _LOGIT_BIAS
            probs = F.softmax(logits, dim=1).squeeze().tolist()
        emotion = _EMOTIONS[int(np.argmax(probs))]
        return {"emotion": emotion, "probs": probs, "emotions": _EMOTIONS}
