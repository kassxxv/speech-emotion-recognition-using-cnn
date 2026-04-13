import os
import numpy as np
import librosa
import pandas as pd


def extract_mel(file_path, n_mels=40, target_frames=200): 
    '''
    Extract Mel spectrogram from audio file, ensuring fixed size.
    n - Number of Mel bands (frequency channels)
    target_frames - Number of time frames to pad/truncate to (4 seconds at 16kHz with hop_length=512)
    '''
    y, sr = librosa.load(file_path, sr=None) # Load audio with original sampling rate

    mel = librosa.feature.melspectrogram( # Extract Mel spectrogram using librosa
        y=y,
        sr=sr,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(mel)
    if mel_db.shape[1] < target_frames:  # Pad with zeros if too short
        pad = target_frames - mel_db.shape[1] # Calculate how many frames to pad
        mel_db = np.pad(mel_db, ((0,0),(0,pad)))
    else:
        mel_db = mel_db[:, :target_frames] # Truncate if too long
    return mel_db

def extract_mfcc(file_path, n_mfcc=40, target_frames=200):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    if mfcc.shape[1] < target_frames: # Pad with zeros if too short
        pad = target_frames - mfcc.shape[1] # Calculate how many frames to pad
        mfcc = np.pad(mfcc, ((0,0),(0,pad)))
    else:
        mfcc = mfcc[:, :target_frames] # Truncate if too long

    return mfcc

def compile_features(csv_path, n_mels=40, feature_root="features"):
    # Read metadata CSV and extract features for all audio files, saving as .npy
    df = pd.read_csv(csv_path)
    total = len(df)

    # Mel folder name includes n_mels so different resolutions don't overwrite each other
    mel_folder  = f"{feature_root}/mel" if n_mels == 40 else f"{feature_root}/mel{n_mels}"
    mfcc_folder = f"{feature_root}/mfcc"

    # Create directories for features if they don't exist
    os.makedirs(mel_folder,  exist_ok=True)
    os.makedirs(mfcc_folder, exist_ok=True)

    print(f"Extracting features: n_mels={n_mels}, saving mel to '{mel_folder}'")

    # Process each file and extract features
    for i, row in enumerate(df.itertuples()):

        wav_path = row.file_path

        mel  = extract_mel(wav_path, n_mels=n_mels)
        mfcc = extract_mfcc(wav_path)

        filename = os.path.basename(wav_path).replace(".wav",".npy")

        np.save(os.path.join(mel_folder,  filename), mel)
        np.save(os.path.join(mfcc_folder, filename), mfcc)
        # Print progress every 500 files
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{total} files...")

    print(f"Feature extraction complete: {total} files processed.")

if __name__ == "__main__":
    compile_features("crema_metadata.csv")