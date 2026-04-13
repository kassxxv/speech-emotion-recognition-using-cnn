"""Shared utilities used by evaluate.py, grad_cam.py, and make_paper_figures.py."""
import numpy as np
import torch
import librosa


def load_compatible_state_dict(model, checkpoint_path, device):
    """Load checkpoint weights into model, skipping keys with shape mismatches."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    model.load_state_dict(compatible, strict=False)
    return skipped


def add_gaussian_noise(y, snr_db):
    """Add zero-mean Gaussian noise to waveform at the given SNR (dB)."""
    signal_power = np.mean(y ** 2)
    snr_linear   = 10 ** (snr_db / 10.0)
    noise_power  = signal_power / snr_linear
    noise        = np.random.randn(len(y)) * np.sqrt(noise_power)
    return (y + noise).astype(y.dtype)


def extract_feature_from_waveform(y, sr, feature_type, n_mels=40, target_frames=200):
    """Extract a fixed-size mel spectrogram or MFCC array from a waveform.

    Args:
        y: waveform array (float32)
        sr: sampling rate
        feature_type: 'mel' or 'mfcc'
        n_mels: number of mel bands / MFCC coefficients (default 40)
        target_frames: pad or truncate to this many time frames (default 200)

    Returns:
        numpy array of shape (n_mels, target_frames), dtype float32
    """
    if feature_type == "mel":
        mel  = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        feat = librosa.power_to_db(mel)
    else:
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mels)

    if feat.shape[1] < target_frames:
        feat = np.pad(feat, ((0, 0), (0, target_frames - feat.shape[1])))
    else:
        feat = feat[:, :target_frames]

    return feat.astype(np.float32)
