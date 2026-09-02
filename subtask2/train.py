##############################################################
# FIRE 2026
# Subtask 3
# Part 1
##############################################################

import os
import random
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import torch

from transformers import AutoTokenizer

##############################################################
# Configuration
##############################################################

MODEL_NAME = "FacebookAI/xlm-roberta-base"

TRAIN_PATH = "../data/subtask2/train.csv"
VALID_PATH = "../data/subtask2/validation.csv"

MAX_LEN = 128
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device :", DEVICE)

##############################################################
# Seed
##############################################################

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

##############################################################
# Load Data
##############################################################

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

train_df = train_df.drop(columns=["Unnamed: 0"], errors="ignore")
val_df = val_df.drop(columns=["Unnamed: 0"], errors="ignore")

print("\nTraining Shape :", train_df.shape)
print("Validation Shape :", val_df.shape)

##############################################################
# Label Encoding
##############################################################

label_encoder = LabelEncoder()

train_df["label"] = label_encoder.fit_transform(
    train_df["Labels"]
)

val_df["label"] = label_encoder.transform(
    val_df["Labels"]
)

NUM_CLASSES = len(label_encoder.classes_)

print("\nClasses :", NUM_CLASSES)
print(label_encoder.classes_)

##############################################################
# Save Encoder
##############################################################

with open("label_encoder.pkl","wb") as f:
    pickle.dump(label_encoder,f)

##############################################################
# Tokenizer
##############################################################

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("\nTraining Samples :", len(train_df))
print("Validation Samples :", len(val_df))
print(train_df["Labels"].value_counts())
##############################################################
# FIRE 2026
# Subtask 3
# Part 2
##############################################################

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoModel

##############################################################
# Dataset
##############################################################

class KannadaDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        text = str(row["Comments"])

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {

            "input_ids":
                encoding["input_ids"].squeeze(),

            "attention_mask":
                encoding["attention_mask"].squeeze(),

            "label":
                torch.tensor(
                    row["label"],
                    dtype=torch.long
                )

        }

##############################################################
# DataLoaders
##############################################################

train_dataset = KannadaDataset(train_df)
val_dataset = KannadaDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

##############################################################
# Model
##############################################################

class KannadaClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME
        )

        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Sequential(

            nn.Linear(hidden, hidden),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(hidden, NUM_CLASSES)

        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask

        )

        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()

        masked = token_embeddings * mask

        pooled = masked.sum(1) / mask.sum(1).clamp(min=1e-9)

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        return logits

##############################################################
# Load Model
##############################################################

model = KannadaClassifier().to(DEVICE)

print(model)

print("\nModel Ready")

##############################################################
# FIRE 2026
# Subtask 3
# Part 3
##############################################################

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import get_linear_schedule_with_warmup

from torch.cuda.amp import autocast, GradScaler

##############################################################
# Hyperparameters
##############################################################

EPOCHS = 10

LR = 1e-5

WEIGHT_DECAY = 0.01

PATIENCE = 3

##############################################################
# Class Weights
##############################################################

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

##############################################################
# Optimizer
##############################################################

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

##############################################################
# Scheduler
##############################################################

total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

##############################################################
# Mixed Precision
##############################################################

##############################################################
# Training
##############################################################

best_macro = 0

early_stop = 0

for epoch in range(EPOCHS):

    print("\n" + "="*70)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("="*70)

    ##########################################################
    # Train
    ##########################################################

    model.train()

    train_loss = 0

    for step, batch in enumerate(train_loader):

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        with autocast():

            logits = model(input_ids, attention_mask)

            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        train_loss += loss.item()
        if step % 50 == 0:
            print(step, loss.item())

    ##########################################################
    # Validation
    ##########################################################

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)

            prediction = torch.argmax(logits, dim=1)

            preds.extend(prediction.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # <-- OUTSIDE the for loop

    acc = accuracy_score(targets, preds)

    macro = f1_score(
        targets,
        preds,
        average="macro"
    )

    weighted = f1_score(
        targets,
        preds,
        average="weighted"
    )

    print(classification_report(
        targets,
        preds,
        digits=4,
        zero_division=0
    ))
    ##########################################################
    # Save Best Model
    ##########################################################

    if macro > best_macro:

        best_macro = macro

        early_stop = 0

        torch.save({

            "model": model.state_dict(),

            "macro_f1": best_macro

        }, "best_subtask3_model.pt")

        print("Best Model Saved")

    else:

        early_stop += 1

        print(f"No Improvement ({early_stop}/{PATIENCE})")

        if early_stop >= PATIENCE:

            print("\nEarly Stopping Activated")

            break

print("\nTraining Completed")

print("Best Macro F1 :", round(best_macro,4))
