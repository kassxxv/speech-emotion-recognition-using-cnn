# Speech Emotion Recognition using CNN

Deep Learning for Speech Emotion Recognition

This repository contains a CNN-based Speech Emotion Recognition (SER) system that classifies audio into 6 emotional categories using Mel-spectrograms and MFCC features extracted with Librosa. The model achieves robust emotion prediction with data augmentation and noise robustness techniques.

## Features
- Mel-spectrogram and MFCC extraction
- CNN with 4 conv layers + attention mechanism
- Robust to noisy audio

## Requirements

Make sure you have the following libraries installed:

- `librosa`
- `numpy`
- `matplotlib`
- `torch`
- `pandas`
- `jupyterlab`
- `scikit-learn`
- `grad-cam`

You can install them using:

```bash
pip -r requirements.txt