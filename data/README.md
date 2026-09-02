# Dataset Information

The datasets used for the **DravidianCodeMix 2026 Shared Task** are not included in this repository.

They should be obtained from the official shared-task organizers and used according to the applicable dataset license and terms of use.

## Expected Directory Structure

After obtaining the datasets, place them locally using the following structure:

```text
data/
│
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

## Subtask 1

Tamil-English Hate Speech dataset.

Required files:

```text
data/subtask1/train.csv
data/subtask1/validation.csv
data/subtask1/test.csv
```

## Subtask 2

Tulu Hope Speech dataset.

Required files:

```text
data/subtask2/train.csv
data/subtask2/validation.csv
data/subtask2/test.csv
```

## Subtask 3

Kannada Hate Speech dataset.

Required files:

```text
data/subtask3/train.csv
data/subtask3/validation.csv
data/subtask3/test.csv
```

## Important

The dataset files are intentionally excluded from Git version control.

Do not upload the datasets to this repository unless redistribution is explicitly permitted by the dataset/shared-task organizers.

The training and prediction scripts expect the corresponding dataset files to be available locally.
