import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

from augmentation import spec_augment

dataset_path = "CREMA-D"
ravdess_path = "RAVDESS"


# CREMA-D emotion mapping (6 emotions)
emotion_map = {
    "ANG": ("angry", 0),
    "DIS": ("disgust", 1),
    "FEA": ("fear", 2),
    "HAP": ("happy", 3),
    "NEU": ("neutral", 4),
    "SAD": ("sad", 5)
}

# RAVDESS emotion mapping — 6 of 8 kept (calm=02 and surprised=08 skipped, no CREMA-D equivalent)
RAVDESS_EMOTION_MAP = {
    "01": ("neutral", 4),
    "03": ("happy",   3),
    "04": ("sad",     5),
    "05": ("angry",   0),
    "06": ("fearful", 2),
    "07": ("disgust", 1),
}


class CREMADataset(Dataset):
    """Dataset class for CREMA-D and RAVDESS emotion recognition."""

    def __init__(self, csv_path, actors, train=False, augment_prob=0.5, feature_type="mel", normalize=False, n_mels=40, feature_root=None): # Initialization
        self.df = pd.read_csv(csv_path) # Load metadata CSV
        self.df = self.df[self.df['actor_id'].isin(actors)] # Filter by actor split
        self.train = train
        self.augment_prob = augment_prob # Probability of applying SpecAugment during training
        self.feature_type = feature_type  # "mel" or "mfcc"
        self.normalize = normalize  # z-score normalization per sample
        self.n_mels = n_mels  # mel band resolution (affects folder name for mel features)
        self.feature_root = feature_root  # override root folder (None = default "features/")
        self.labels = self.df["emotion_id"].tolist() # Store labels for weighted sampling

    def __len__(self):
        return len(self.df)

    def _feature_folder(self):
        """Return the folder path for the current feature type and n_mels setting."""
        root = self.feature_root if self.feature_root else "features"
        if self.feature_type == "mel" and self.n_mels != 40:
            return f"{root}/mel{self.n_mels}"
        return f"{root}/{self.feature_type}"

    def __getitem__(self, idx): # Get item by index
        row = self.df.iloc[idx] # Get file path and corresponding feature
        wav_path = row["file_path"] # Path to the original wav file
        filename = os.path.basename(wav_path).replace(".wav", ".npy")  # Other format

        feature = np.load(os.path.join(self._feature_folder(), filename))

        # Z-score normalization per sample — removes loudness bias, helps CNN focus on shape
        if self.normalize:
            feature = (feature - feature.mean()) / (feature.std() + 1e-8)

        # Apply SpecAugment during training
        if self.train and np.random.random() < self.augment_prob: # Randomly apply augmentation to increase robustness
            feature = spec_augment(feature, freq_mask_param=10, time_mask_param=30,
                                   num_freq_masks=2, num_time_masks=2) # More aggressive augmentation for better generalization

        # Convert to tensor with channel dimension (1, n_mels, 200)
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


def generate_ravdess_metadata():
    """Parse RAVDESS filenames and create ravdess_metadata.csv.

    Filename format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Only speech modality (03) is kept. Calm (02) and surprised (08) are skipped.
    Scans top-level Actor_* dirs only — audio_speech_actors_01-24/ is a duplicate mirror.
    """
    data = []
    actor_dirs = sorted([
        d for d in os.listdir(ravdess_path)
        if d.startswith("Actor_") and os.path.isdir(os.path.join(ravdess_path, d))
    ])
    for actor_dir in actor_dirs:
        actor_path = os.path.join(ravdess_path, actor_dir)
        for file in os.listdir(actor_path):
            if not file.endswith(".wav"):
                continue
            parts = file.replace(".wav", "").split("-")
            if len(parts) != 7:
                continue
            modality, vocal_channel, emotion_code, intensity, statement, repetition, actor_field = parts
            if modality != "03":  # speech only
                continue
            if emotion_code not in RAVDESS_EMOTION_MAP:
                continue  # skip calm=02, surprised=08
            emotion_name, emotion_id = RAVDESS_EMOTION_MAP[emotion_code]
            data.append({
                "actor_id":     int(actor_field),
                "emotion_code": emotion_code,
                "emotion_name": emotion_name,
                "emotion_id":   emotion_id,
                "intensity":    intensity,
                "statement":    statement,
                "file_path":    os.path.join(actor_path, file),
            })
    df = pd.DataFrame(data)
    df.to_csv("ravdess_metadata.csv", index=False)
    print(f"Created ravdess_metadata.csv with {len(df)} samples")
    return df


# Generate CSVs if not exists
if not os.path.exists("crema_metadata.csv"):
    generate_crema_metadata()
if not os.path.exists("ravdess_metadata.csv") and os.path.exists(ravdess_path):
    generate_ravdess_metadata()

# Get unique actor IDs
df_temp = pd.read_csv("crema_metadata.csv")
all_actors = sorted(df_temp['actor_id'].unique()) # Returns sorted list of unique actor IDs
n_actors = len(all_actors)

# Split: 70% train / 15% val / 15% test (by actor for speaker-independent evaluation)
rng = np.random.default_rng(42) # Random number generator with standard seed for reproducibility
shuffled_actors = rng.permutation(all_actors) # Shuffle actor IDs
n_train = int(n_actors * 0.70)
n_val   = int(n_actors * 0.15)
train_actors = shuffled_actors[:n_train]                   # 70% for training
val_actors   = shuffled_actors[n_train:n_train + n_val]    # 15% for validation
test_actors  = shuffled_actors[n_train + n_val:]           # 15% held-out test set

print(f"Train actors: {len(train_actors)}, Val actors: {len(val_actors)}, Test actors: {len(test_actors)}")


def get_ravdess_test_actors():
    """Return the held-out test actor IDs for RAVDESS (same 70/15/15 split logic)."""
    df = pd.read_csv("ravdess_metadata.csv")
    all_r = sorted(df["actor_id"].unique())
    rng   = np.random.default_rng(42)
    sh    = rng.permutation(all_r)
    n_tr  = int(len(all_r) * 0.70)
    n_vl  = int(len(all_r) * 0.15)
    return sh[n_tr + n_vl:]


def get_loaders(feature_type="mel", augment_prob=0.7, batch_size=32, normalize=False, n_mels=40, dataset="crema"):
    """Create train and validation DataLoaders for the given feature type and dataset.

    Args:
        feature_type: "mel" or "mfcc" — selects features/{feature_type}/ folder
        augment_prob: SpecAugment probability during training (0.0 = no augmentation)
        batch_size: batch size for both loaders
        normalize: apply z-score normalization per sample
        n_mels: number of mel bands (loads from features/mel{n_mels}/ when != 40)
        dataset: "crema" or "ravdess" — selects metadata CSV and feature folder root
    """
    if dataset == "ravdess":
        csv_path     = "ravdess_metadata.csv"
        feature_root = "features/ravdess"
        df_r  = pd.read_csv(csv_path)
        all_r = sorted(df_r["actor_id"].unique())
        rng_r = np.random.default_rng(42)
        sh_r  = rng_r.permutation(all_r)
        n_tr  = int(len(all_r) * 0.70)
        n_vl  = int(len(all_r) * 0.15)
        t_act = sh_r[:n_tr]
        v_act = sh_r[n_tr:n_tr + n_vl]
    else:
        csv_path     = "crema_metadata.csv"
        feature_root = None  # CREMADataset default (features/)
        t_act = train_actors
        v_act = val_actors

    train_ds = CREMADataset(
        csv_path,
        t_act,
        train=True,
        augment_prob=augment_prob,
        feature_type=feature_type,
        normalize=normalize,
        n_mels=n_mels,
        feature_root=feature_root,
    )
    val_ds = CREMADataset(
        csv_path,
        v_act,
        train=False,
        feature_type=feature_type,
        normalize=normalize,
        n_mels=n_mels,
        feature_root=feature_root,
    )

    # Balanced sampling to improve macro-F1 on minority emotions
    labels = np.array(train_ds.labels)
    class_counts = np.bincount(labels, minlength=6)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    t_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    v_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    return t_loader, v_loader


