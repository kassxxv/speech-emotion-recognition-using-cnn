# Speech Emotion Recognition using CNN

CNN-based classification of 6 emotions from audio using Mel-spectrogram and MFCC features on the CREMA-D dataset.

## Setup

1. Clone the repo
2. Download [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) → place `.wav` files in `CREMA-D/`
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Reproduce Experiments

```bash
# Extract features (run once)
python feature_extraction.py

# Train
python train.py --feature mel
python train.py --feature mfcc
python train.py --feature mel --no-augment   # ablation: no SpecAugment

# Evaluate (confusion matrix + noise robustness)
python evaluate.py --feature mel
python evaluate.py --feature mfcc

# Grad-CAM (all 6 emotions)
python grad_cam.py --feature mel
python grad_cam.py --feature mfcc
```

Then open `notebooks/` in Jupyter to explore results.

## Emotions

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad`
