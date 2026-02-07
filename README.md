# Brain Tumor MRI Binary Classification Pipeline

This repository provides a minimal PyTorch training pipeline for binary classification of brain tumor MRI images.

## Dataset layout

Organize your MRI images with `torchvision.datasets.ImageFolder` convention:

```
data/
  no_tumor/
    img1.png
  tumor/
    img2.png
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --data-dir data --epochs 15 --batch-size 16
```

The script saves the trained model to `artifacts/model.pt` by default.
