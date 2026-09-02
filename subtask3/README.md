# Subtask 3 – Kannada Hate Speech

## Overview

This directory contains the training and prediction implementation for **Subtask 3 of DravidianCodeMix 2026**.

The task performs binary hate speech classification for Kannada text.

## Model

The system uses **XLM-RoBERTa Base** as the multilingual Transformer encoder.

The architecture consists of a shared Transformer representation followed by a binary classification head.

```text
XLM-RoBERTa Base
        │
        ▼
Shared Text Representation
        │
        ▼
Classification Head
        │
        ├── Class 0
        └── Class 1
```

## Files

```text
subtask3/
├── train.py
├── predict.py
└── README.md
```

### `train.py`

Trains the Kannada hate speech classifier using the training and validation datasets.

### `predict.py`

Loads the trained model and generates predictions for the test dataset.

## Dataset

The dataset is not included in this repository.

Place the required files locally under:

```text
data/subtask3/
├── train.csv
├── validation.csv
└── test.csv
```

Refer to `data/README.md` for additional information.

## Training

From the repository root:

```bash
python subtask3/train.py
```

The training process uses CUDA when available.

## Prediction

After training:

```bash
python subtask3/predict.py
```

The prediction script generates predictions for the test instances.

## Official Results

The team achieved **3rd place** in Subtask 3.

| Metric       |      Score |
| ------------ | ---------: |
| Precision    |     0.7653 |
| Recall       |     0.7679 |
| **Macro-F1** | **0.7652** |

**Official Rank:** 3

## Development Results

The best validation experiment achieved:

* Accuracy: 0.7480
* Macro-F1: 0.7407

These are development/validation results and are separate from the official leaderboard results.

## Reproducibility

The implementation uses Python, PyTorch, and Hugging Face Transformers with XLM-RoBERTa Base.

The shared-task datasets and trained model checkpoints are not included in this repository.

Users should obtain the required dataset from the official organizers before running the training or prediction scripts.
