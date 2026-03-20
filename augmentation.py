"""Minimal spectrogram augmentation used by the training dataloader."""

import numpy as np
import torch


def _random_mask(length, max_width):
    """Return (start, width) for a random contiguous mask."""
    width_limit = min(max_width, length)
    if width_limit <= 0:
        return 0, 0

    width = np.random.randint(0, width_limit + 1)
    if width == 0:
        return 0, 0

    start = np.random.randint(0, length - width + 1)
    return start, width


def spec_augment(spec, freq_mask_param=8, time_mask_param=25, num_freq_masks=1, num_time_masks=1):
    """Apply frequency/time masking to a mel spectrogram.

    Accepts shapes (n_mels, time) or (1, n_mels, time) and returns the same type/shape.
    """
    is_tensor = isinstance(spec, torch.Tensor)
    x = spec.clone() if is_tensor else np.array(spec, copy=True)

    squeeze_channel = x.ndim == 3
    if squeeze_channel:
        x = x.squeeze(0)

    n_mels, n_frames = x.shape

    for _ in range(num_freq_masks):
        start, width = _random_mask(n_mels, freq_mask_param)
        if width > 0:
            x[start:start + width, :] = 0

    for _ in range(num_time_masks):
        start, width = _random_mask(n_frames, time_mask_param)
        if width > 0:
            x[:, start:start + width] = 0

    if squeeze_channel:
        x = x.unsqueeze(0) if is_tensor else np.expand_dims(x, axis=0)

    return x
