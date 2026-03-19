"""
Data augmentation for speech emotion recognition.

Two types of augmentation:
1. SpecAugment - Applied on spectrograms during training (fast, no I/O)
2. Audio augmentation - Applied on raw audio before feature extraction
"""

import numpy as np
import torch


# =============================================================================
# SpecAugment (on spectrograms) - Recommended for training
# =============================================================================

def spec_augment(spec, freq_mask_param=8, time_mask_param=25,
                 num_freq_masks=1, num_time_masks=1):
    """
    Apply SpecAugment to a spectrogram.

    Args:
        spec: Spectrogram tensor of shape (1, n_mels, time) or (n_mels, time)
        freq_mask_param: Maximum frequency mask width
        time_mask_param: Maximum time mask width
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply

    Returns:
        Augmented spectrogram
    """
    # Handle both (1, n_mels, time) and (n_mels, time) shapes
    if isinstance(spec, torch.Tensor):
        spec = spec.clone()
        squeeze = False
        if spec.dim() == 3:
            spec = spec.squeeze(0)
            squeeze = True
        n_mels, n_frames = spec.shape

        # Frequency masking
        for _ in range(num_freq_masks):
            f = np.random.randint(0, min(freq_mask_param, n_mels))
            f0 = np.random.randint(0, n_mels - f)
            spec[f0:f0+f, :] = 0

        # Time masking
        for _ in range(num_time_masks):
            t = np.random.randint(0, min(time_mask_param, n_frames))
            t0 = np.random.randint(0, n_frames - t)
            spec[:, t0:t0+t] = 0

        if squeeze:
            spec = spec.unsqueeze(0)
    else:
        # NumPy array
        spec = spec.copy()
        n_mels, n_frames = spec.shape

        for _ in range(num_freq_masks):
            f = np.random.randint(0, min(freq_mask_param, n_mels))
            f0 = np.random.randint(0, n_mels - f)
            spec[f0:f0+f, :] = 0

        for _ in range(num_time_masks):
            t = np.random.randint(0, min(time_mask_param, n_frames))
            t0 = np.random.randint(0, n_frames - t)
            spec[:, t0:t0+t] = 0

    return spec


def time_warp(spec, W=5):
    """
    Apply time warping to spectrogram (simplified version).
    Randomly shifts a portion of the spectrogram in time.

    Args:
        spec: Spectrogram of shape (n_mels, time)
        W: Maximum warp distance

    Returns:
        Warped spectrogram
    """
    if isinstance(spec, torch.Tensor):
        spec = spec.clone()
    else:
        spec = spec.copy()

    n_frames = spec.shape[-1]
    if n_frames <= 2 * W:
        return spec

    # Random source and destination points
    src = np.random.randint(W, n_frames - W)
    dest = src + np.random.randint(-W, W + 1)
    dest = max(W, min(dest, n_frames - W))

    # Simple shift (approximate warping)
    if dest != src:
        if dest > src:
            spec[..., src:dest] = spec[..., src:src+1].repeat(
                1 if spec.ndim == 2 else spec.shape[0], dest - src
            ).reshape(spec[..., src:dest].shape)
        else:
            spec[..., dest:src] = spec[..., src:src+1].repeat(
                1 if spec.ndim == 2 else spec.shape[0], src - dest
            ).reshape(spec[..., dest:src].shape)

    return spec


# =============================================================================
# Audio augmentation (on raw waveforms) - For expanding dataset
# =============================================================================

def add_noise(y, noise_level=0.005):
    """Add Gaussian noise to audio signal."""
    noise = np.random.randn(len(y)) * noise_level
    return y + noise


def time_stretch(y, rate=None):
    """
    Time stretch audio without changing pitch.
    Requires librosa.
    """
    import librosa
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y, sr, n_steps=None):
    """
    Shift pitch without changing tempo.
    Requires librosa.
    """
    import librosa
    if n_steps is None:
        n_steps = np.random.randint(-3, 4)  # -3 to +3 semitones
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def random_gain(y, min_gain=0.7, max_gain=1.3):
    """Apply random gain (volume change)."""
    gain = np.random.uniform(min_gain, max_gain)
    return y * gain


def augment_audio(y, sr, p=0.5):
    """
    Apply random augmentations to audio signal.

    Args:
        y: Audio signal
        sr: Sample rate
        p: Probability of each augmentation

    Returns:
        Augmented audio signal
    """
    if np.random.random() < p:
        y = add_noise(y, noise_level=np.random.uniform(0.001, 0.01))

    if np.random.random() < p:
        y = time_stretch(y, rate=np.random.uniform(0.85, 1.15))

    if np.random.random() < p:
        y = pitch_shift(y, sr, n_steps=np.random.randint(-2, 3))

    if np.random.random() < p:
        y = random_gain(y, min_gain=0.8, max_gain=1.2)

    return y
