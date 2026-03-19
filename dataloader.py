import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from augmentation import spec_augment

dataset_path = "RAVDESS"


class RAVDESSDataset(Dataset):
    """Custom dataset class for RAVDESS emotion recognition."""

    def __init__(self, csv_path, actors, train=False, augment_prob=0.6):
        """
        Args:
            csv_path: Path to metadata CSV
            actors: List of actor IDs to include
            train: If True, apply data augmentation
            augment_prob: Probability of applying augmentation (0.0-1.0)
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['actor_id'].isin(actors)]
        self.train = train
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row["file_path"]

        # Load mel spectrogram
        filename = os.path.basename(wav_path).replace(".wav", ".npy")
        feature_path = os.path.join("features/mel", filename)
        feature = np.load(feature_path)

        # Apply SpecAugment during training (balanced augmentation)
        if self.train and np.random.random() < self.augment_prob:
            feature = spec_augment(
                feature,
                freq_mask_param=8,
                time_mask_param=25,
                num_freq_masks=1,
                num_time_masks=2
            )

        # Convert to tensor with channel dimension
        feature = torch.tensor(feature).float().unsqueeze(0)

        # Labels: 1-8 -> 0-7
        label = row["emotion_id"] - 1

        return feature, label

train_dataset = RAVDESSDataset(
    "ravdess_metadata.csv",
    list(range(1, 17)),
    train=True,
    augment_prob=0.5  # Balanced augmentation
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_dataset = RAVDESSDataset(
    "ravdess_metadata.csv",
    list(range(17, 25)),
    train=False  # No augmentation for validation
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# dictionary for IDs
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# filename.wav pattern: Modality, Vocal Channel, Emotion, Intensity, Statement, Repetition, Actor
pattern = re.compile(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav")

data = []

'''
"os.walk": Walks through the dataset directory, matches files with the specified pattern,
extracts metadata from the filename, and compiles it into a DataFrame which is then saved as a CSV file.
'''
for root, dirs, files in os.walk(dataset_path): 
    for file in files:
        if file.endswith(".wav"):
            match = pattern.match(file)
            if match:
                modality = int(match.group(1))
                vocal_channel = int(match.group(2))
                emotion_id = int(match.group(3))
                intensity = int(match.group(4))
                statement = int(match.group(5))
                repetition = int(match.group(6))
                actor_id = int(match.group(7))

                file_path = os.path.join(root, file)

                actor_gender = "male" if actor_id % 2 == 1 else "female"

                data.append({
                    "modality": modality,
                    "vocal_channel": vocal_channel,
                    "emotion_id": emotion_id,
                    "emotion_name": emotion_map[f"{emotion_id:02}"],
                    "intensity": intensity,
                    "statement": statement,
                    "repetition": repetition,
                    "actor_id": actor_id,
                    "actor_gender": actor_gender,
                    "file_path": file_path
                })

df = pd.DataFrame(data)
df.to_csv("ravdess_metadata.csv", index=False)
print(df.head())