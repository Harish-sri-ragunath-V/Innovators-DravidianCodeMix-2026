# Subtask 1 – Tamil-English Hate Speech

## Overview

This directory contains the training and prediction implementation for **Subtask 1 of DravidianCodeMix 2026**.

The task focuses on hate speech classification in Tamil-English code-mixed text using a hierarchical annotation scheme.

## Model

The system uses **XLM-RoBERTa Large** as the multilingual Transformer encoder.

A hierarchical classification architecture is used with five classification levels.

```text
XLM-RoBERTa Large
        │
        ▼
Shared Text Representation
        │
        ├── Level 1 → 3 classes
        ├── Level 2 → 6 classes
        ├── Level 3 → 3 classes
        ├── Level 4 → 32 classes
        └── Level 5 → 7 classes
```

## Files

```text
subtask1/
├── train.py
├── predict.py
└── README.md
```

### `train.py`

Trains the hierarchical XLM-RoBERTa classifier using the training and validation datasets.

### `predict.py`

Loads the trained model and generates predictions for the test dataset.

## Dataset

The dataset is not included in this repository.

Place the required files locally under:

```text
data/subtask1/
├── train.csv
├── validation.csv
└── test.csv
```

Refer to `data/README.md` for more information.

## Training

From the repository root:

```bash
python subtask1/train.py
```

The training process uses GPU acceleration when CUDA is available.

## Prediction

After training the model:

```bash
python subtask1/predict.py
```

The prediction script generates predictions for the test instances.

## Official Results

The team achieved **2nd place** in Subtask 1.

| Level                |   Macro-F1 |
| -------------------- | ---------: |
| Level 1              |     0.5403 |
| Level 2              |     0.5231 |
| Level 3              |     0.6729 |
| Level 4              |     0.1048 |
| Level 5              |     0.3560 |
| **Overall Macro-F1** | **0.4394** |

**Official Rank:** 2

## Development Results

The best validation result during our training experiments was:

* Level 4 Accuracy: 0.5075
* Level 4 Macro-F1: 0.1162

These are development/validation results and are separate from the official leaderboard results.

## Reproducibility

The model uses the Hugging Face Transformers implementation of XLM-RoBERTa Large.

The exact dataset files and trained model checkpoints are not included in this repository.

Users should obtain the shared-task dataset from the official organizers before running the training or prediction scripts.
