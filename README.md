# Innovators @ DravidianCodeMix 2026

This repository contains the implementation and experimental results of **Team Innovators** for the **DravidianCodeMix 2026 Shared Task** under FIRE 2026.

The team participated in three task settings involving hate speech and hope speech detection in Dravidian languages.

## Team

**Team Name:** Innovators

**Authors:**
* Shudapreyaa R S
* Harish Sri Ragunath V
* Jaiyanth T
* Hari Prasath G

## Tasks

| Subtask   | Language / Task           | Approach                       |
| --------- | ------------------------- | ------------------------------ |
| Subtask 1 | Tamil-English Hate Speech | Hierarchical XLM-RoBERTa Large |
| Subtask 2 | Tulu Hope Speech          | Multi-task XLM-RoBERTa Large   |
| Subtask 3 | Kannada Hate Speech       | XLM-RoBERTa Base               |

## Subtask 1 – Tamil-English Hate Speech

A hierarchical classification model based on **XLM-RoBERTa Large** was used to predict five annotation levels.

The model contains separate classification heads for:

* Level 1 – 3 classes
* Level 2 – 6 classes
* Level 3 – 3 classes
* Level 4 – 32 classes
* Level 5 – 7 classes

### Official Result

* **Overall Macro-F1:** 0.4394
* **Rank:** 2

| Level       |   Macro-F1 |
| ----------- | ---------: |
| Level 1     |     0.5403 |
| Level 2     |     0.5231 |
| Level 3     |     0.6729 |
| Level 4     |     0.1048 |
| Level 5     |     0.3560 |
| **Overall** | **0.4394** |

## Subtask 2 – Tulu Hope Speech

A multi-task classification architecture based on **XLM-RoBERTa Large** was used.

The model jointly predicts:

* Coarse tone – 3 classes
* Fine hope type – 5 classes

### Official Result

* **Overall Macro-F1:** 0.4348
* **Rank:** 1

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        |     0.5121 |
| Macro Precision |     0.4386 |
| Macro Recall    |     0.4377 |
| Macro-F1        | **0.4348** |
| Weighted-F1     |     0.5210 |

## Subtask 3 – Kannada Hate Speech

A binary hate-speech classifier based on **XLM-RoBERTa Base** was used.

### Official Result

* **Macro-F1:** 0.7652
* **Rank:** 3

| Metric    |      Score |
| --------- | ---------: |
| Precision |     0.7653 |
| Recall    |     0.7679 |
| Macro-F1  | **0.7652** |

## Repository Structure

```text
Innovators-DravidianCodeMix-2026/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── subtask1/
│   ├── train.py
│   ├── predict.py
│   └── README.md
│
├── subtask2/
│   ├── train.py
│   ├── predict.py
│   └── README.md
│
├── subtask3/
│   ├── train.py
│   ├── predict.py
│   └── README.md
│
└── results/
    ├── subtask1_results.txt
    ├── subtask2_results.txt
    └── subtask3_results.txt
```

## Dataset

The DravidianCodeMix 2026 datasets are **not included in this repository**.

Due to dataset distribution and usage restrictions, users should obtain the datasets from the official shared-task organizers and place them locally according to the structure described in:

```text
data/README.md
```

The expected local structure is:

```text
data/
├── subtask1/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── subtask2/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
└── subtask3/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

The CSV files should **not be committed to this repository** unless redistribution is explicitly permitted by the organizers.

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/Harish-sri-ragunath-V/Innovators-DravidianCodeMix-2026
cd Innovators-DravidianCodeMix-2026
pip install -r requirements.txt
```

## Running the Experiments

### Subtask 1

```bash
python subtask1/train.py
```

For prediction:

```bash
python subtask1/predict.py
```

### Subtask 2

```bash
python subtask2/train.py
```

For prediction:

```bash
python subtask2/predict.py
```

### Subtask 3

```bash
python subtask3/train.py
```

For prediction:

```bash
python subtask3/predict.py
```

## Hardware

The experiments were performed using GPU acceleration with CUDA.

The implementation automatically uses CUDA when available and falls back to CPU otherwise.

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* XLM-RoBERTa
* pandas
* NumPy
* scikit-learn

## Reproducibility

The repository provides training and prediction scripts for all three subtasks.

Model checkpoints and label encoder files are not included in the repository by default. The training scripts can be used to reproduce the models after obtaining the corresponding shared-task datasets.

## Results

Detailed experimental results are available in:

```text
results/subtask1_results.txt
results/subtask2_results.txt
results/subtask3_results.txt
```

## Paper

The team's working note for FIRE 2026 / DravidianCodeMix 2026 describes the methodology, experiments, results, and analysis.

The final paper link will be added after publication/submission.

## Code Repository

The source code, training scripts, prediction scripts, and experimental results for all three subtasks are available at:

[GitHub Repository](https://github.com/Harish-sri-ragunath-V/Innovators-DravidianCodeMix-2026)

## License

This repository contains research code developed for participation in the DravidianCodeMix 2026 shared task.

Dataset and other third-party resources remain subject to their respective licenses and usage conditions.
