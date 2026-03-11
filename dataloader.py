import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

dataset_path = "RAVDESS"

class RAVDESSDataset(Dataset): # Custom dataset class for RAVDESS, from pyTorch

    def __init__(self, csv_path, actors):

        self.df = pd.read_csv(csv_path)

        # Filter the DataFrame to include only rows where 'actor_id' is in the specified list of actors
        self.df = self.df[self.df['actor_id'].isin(actors)] 

    def __len__(self):
        return len(self.df)
    
    # Retrieves the feature and label for a given index in the dataset
    def __getitem__(self, idx):

        row = self.df.iloc[idx] # iloc is used to access a row by its integer index

        wav_path = row["file_path"] # Returns file path for index

        # Extracts filename and replaces .wav with .npy to get the corresponding feature file path
        filename = os.path.basename(wav_path).replace(".wav", ".npy") 
        feature_path = os.path.join("features/mel", filename) # Constructs the full path to the feature file
        feature = np.load(feature_path)
        # Converts the feature to a PyTorch tensor, normalizes it, and adds a channel dimension (unsqueeze(0))
        feature = torch.tensor(feature).float().unsqueeze(0)
        # Changes the label from 1-8 to 0-7 by subtracting 1, since PyTorch expects labels to start from 0
        label = row["emotion_id"] - 1

        return feature, label

train_dataset = RAVDESSDataset(
    "ravdess_metadata.csv",
    list(range(1,17))
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32, # Number of samples per batch to load
    shuffle=True
)
val_dataset = RAVDESSDataset(
    "ravdess_metadata.csv",
    list(range(17,25))
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