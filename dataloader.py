import os
import re
import pandas as pd

dataset_path = "RAVDESS"

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

# filename.wav
pattern = re.compile(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav")

data = []

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