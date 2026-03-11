import os
import numpy as np
import librosa
import pandas as pd


def extract_mel(file_path, n_mels=40, target_frames=200):

    y, sr = librosa.load(file_path, sr=None)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(mel)

    if mel_db.shape[1] < target_frames:
        pad = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)))
    else:
        mel_db = mel_db[:, :target_frames]

    return mel_db

def extract_mfcc(file_path, n_mfcc=40, target_frames=200):
    
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    if mfcc.shape[1] < target_frames:
        pad = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)))
    else:
        mfcc = mfcc[:, :target_frames]

    return mfcc

def compile_features(csv_path):

    df = pd.read_csv(csv_path)

    os.makedirs("features/mel", exist_ok=True)
    os.makedirs("features/mfcc", exist_ok=True)


    for _, row in df.iterrows():

        wav_path = row["file_path"]

        mel = extract_mel(wav_path)
        mfcc = extract_mfcc(wav_path)


        filename = os.path.basename(wav_path).replace(".wav",".npy")

        save_path = os.path.join("features/mel", filename)

        np.save(save_path, mel)

        save_path = os.path.join("features/mfcc", filename)

        np.save(save_path, mfcc)

    print("Feature compilation finished.")

if __name__ == "__main__":
    compile_features("ravdess_metadata.csv")