# Subtask 2 – Tulu Hope Speech

## Overview

This directory contains the training and prediction implementation for **Subtask 2 of DravidianCodeMix 2026**.

The task focuses on hope speech detection in Tulu using coarse- and fine-grained labels.

## Model

The system uses **XLM-RoBERTa Large** as the multilingual Transformer encoder.

A multi-task architecture jointly predicts the coarse and fine labels.

```text
XLM-RoBERTa Large
        │
        ▼
Shared Representation
        │
        ├── Coarse Head → 3 classes
        │
        └── Fine Head → 5 classes
```

## Files

```text
subtask2/
├── train.py
├── predict.py
└── README.md
```

### `train.py`

Trains the multi-task hope speech classifier using the training and validation datasets.

### `predict.py`

Loads the trained model and generates predictions for the test dataset.

## Dataset

The dataset is not included in this repository.

Place the required files locally under:

```text
data/subtask2/
├── train.csv
├── validation.csv
└── test.csv
```

Refer to `data/README.md` for dataset information.

## Training

From the repository root:

```bash
python subtask2/train.py
```

The training process uses CUDA when available.

## Prediction

After training:

```bash
python subtask2/predict.py
```

The prediction script generates predictions for the test instances.

## Official Results

The team achieved **1st place** in Subtask 2.

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        |     0.5121 |
| Macro Precision |     0.4386 |
| Macro Recall    |     0.4377 |
| **Macro-F1**    | **0.4348** |
| Weighted-F1     |     0.5210 |

**Official Rank:** 1

## Development Results

The best validation experiment achieved:

* Fine Accuracy: 0.4900
* Fine Macro-F1: 0.4356
* Official Joint Score on validation: 0.4892

These values are development/validation results and should not be confused with the official leaderboard result.

## Reproducibility

The implementation uses Python, PyTorch, and Hugging Face Transformers.

The shared-task datasets and trained model checkpoints are not included in this repository.

Users should obtain the required data from the official organizers before running the training or prediction scripts.
