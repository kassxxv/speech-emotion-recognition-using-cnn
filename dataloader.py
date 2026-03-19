import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from augmentation import spec_augment

dataset_path = "CREMA-D"

# CREMA-D emotion mapping (6 emotions)
emotion_map = {
    "ANG": ("angry", 0),
    "DIS": ("disgust", 1),
    "FEA": ("fear", 2),
    "HAP": ("happy", 3),
    "NEU": ("neutral", 4),
    "SAD": ("sad", 5)
}


class CREMADataset(Dataset):
    """Dataset class for CREMA-D emotion recognition."""

    def __init__(self, csv_path, actors, train=False, augment_prob=0.5):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['actor_id'].isin(actors)]
        self.train = train
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row["file_path"]
        filename = os.path.basename(wav_path).replace(".wav", ".npy")

        # Load mel spectrogram
        feature = np.load(os.path.join("features/mel", filename))

        # Apply SpecAugment during training
        if self.train and np.random.random() < self.augment_prob:
            feature = spec_augment(feature, freq_mask_param=8, time_mask_param=25,
                                   num_freq_masks=1, num_time_masks=2)

        # Convert to tensor with channel dimension (1, 40, 200)
        feature = torch.tensor(feature).float().unsqueeze(0)

        label = row["emotion_id"]

        return feature, label


# Generate metadata CSV
def generate_crema_metadata():
    """Parse CREMA-D filenames and create metadata CSV."""
    # Pattern: ActorID_Statement_Emotion_Level.wav
    pattern = re.compile(r"(\d+)_(\w+)_(\w+)_(\w+)\.wav")

    data = []
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            match = pattern.match(file)
            if match:
                actor_id = int(match.group(1))
                statement = match.group(2)
                emotion_code = match.group(3)
                intensity = match.group(4)

                if emotion_code in emotion_map:
                    emotion_name, emotion_id = emotion_map[emotion_code]
                    file_path = os.path.join(dataset_path, file)

                    data.append({
                        "actor_id": actor_id,
                        "statement": statement,
                        "emotion_code": emotion_code,
                        "emotion_name": emotion_name,
                        "emotion_id": emotion_id,
                        "intensity": intensity,
                        "file_path": file_path
                    })

    df = pd.DataFrame(data)
    df.to_csv("crema_metadata.csv", index=False)
    print(f"Created crema_metadata.csv with {len(df)} samples")
    return df


# Generate CSV if not exists
if not os.path.exists("crema_metadata.csv"):
    generate_crema_metadata()

# Get unique actor IDs
df_temp = pd.read_csv("crema_metadata.csv")
all_actors = sorted(df_temp['actor_id'].unique())
n_actors = len(all_actors)

# Split: ~80% train, ~20% val (by actor for speaker-independent evaluation)
train_actors = all_actors[:int(n_actors * 0.8)]  # First 72 actors
val_actors = all_actors[int(n_actors * 0.8):]    # Last 19 actors

print(f"Train actors: {len(train_actors)}, Val actors: {len(val_actors)}")

train_dataset = CREMADataset(
    "crema_metadata.csv",
    train_actors,
    train=True,
    augment_prob=0.5
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_dataset = CREMADataset(
    "crema_metadata.csv",
    val_actors,
    train=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
