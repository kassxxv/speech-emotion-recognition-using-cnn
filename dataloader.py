import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

    def __init__(self, csv_path, actors, train=False, augment_prob=0.5): # Initialization
        self.df = pd.read_csv(csv_path) # Load metadata CSV
        self.df = self.df[self.df['actor_id'].isin(actors)] # Filter by actor split
        self.train = train 
        self.augment_prob = augment_prob # Probability of applying SpecAugment during training
        self.labels = self.df["emotion_id"].tolist() # Store labels for weighted sampling

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx): # Get item by index
        row = self.df.iloc[idx] # Get file path and corresponding mel spectrogram feature
        wav_path = row["file_path"] # Path to the original wav file
        filename = os.path.basename(wav_path).replace(".wav", ".npy")  # Other format

        # Load mel spectrogram only (1-channel)
        feature = np.load(os.path.join("features/mel", filename)) # Temporary only mel spectrogram

        # Apply SpecAugment during training
        if self.train and np.random.random() < self.augment_prob: # Randomly apply augmentation to increase robustness
            feature = spec_augment(feature, freq_mask_param=10, time_mask_param=30,
                                   num_freq_masks=2, num_time_masks=2) # More aggressive augmentation for better generalization

        # Convert to tensor with channel dimension (1, 40, 200)
        feature = torch.tensor(feature).float().unsqueeze(0)

        label = row["emotion_id"] # Get emotion label as integer

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
all_actors = sorted(df_temp['actor_id'].unique()) # Returns sorted list of unique actor IDs
n_actors = len(all_actors)

# Split: ~80% train, ~20% val (by actor for speaker-independent evaluation)
rng = np.random.default_rng(42) # Random number generator with standard seed for reproducibility
shuffled_actors = rng.permutation(all_actors) # Shuffle actor IDs
train_actors = shuffled_actors[:int(n_actors * 0.8)] # First 80% of shuffled actors for training
val_actors = shuffled_actors[int(n_actors * 0.8):] # Remaining 20% of shuffled actors for validation

print(f"Train actors: {len(train_actors)}, Val actors: {len(val_actors)}")

train_dataset = CREMADataset(
    "crema_metadata.csv",
    train_actors,
    train=True, # Use training set with augmentation
    augment_prob=0.7  # Increased for more regularization
)

val_dataset = CREMADataset(
    "crema_metadata.csv",
    val_actors,
    train=False # Use validation set without augmentation
)

# Balanced sampling to improve macro-F1 on minority emotions.
train_labels = np.array(train_dataset.labels)
class_counts = np.bincount(train_labels, minlength=6)
class_weights = 1.0 / np.maximum(class_counts, 1)
sample_weights = class_weights[train_labels]
train_sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=train_sampler # Use weighted random sampler for balanced batches
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
