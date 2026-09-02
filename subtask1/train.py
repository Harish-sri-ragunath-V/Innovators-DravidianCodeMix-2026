import os
import random
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW

# Configuration
TRAIN_PATH = "../data/subtask1/train.csv"
VALID_PATH = "../data/subtask1/validation.csv"

MODEL_NAME = "FacebookAI/xlm-roberta-large"

MAX_LEN = 256

TRAIN_BATCH_SIZE = 8

VALID_BATCH_SIZE = 16

EPOCHS = 8

LR = 1e-5

WEIGHT_DECAY = 0.01

DROPOUT = 0.3

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

print("Device :", DEVICE)

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

seed_everything()


train_df = pd.read_csv(TRAIN_PATH)

valid_df = pd.read_csv(VALID_PATH)

print(train_df.shape)

print(valid_df.shape)


train_df = train_df.dropna().reset_index(drop=True)

valid_df = valid_df.dropna().reset_index(drop=True)


encoders = {}

columns = [

    ("Level1_Class","l1"),

    ("Level2_Target","l2"),

    ("Level3_Category","l3"),

    ("Level4_Subcategory","l4"),

    ("Level5_Bias","l5")

]

for original,new in columns:

    encoder = LabelEncoder()

    train_df[new] = encoder.fit_transform(
        train_df[original]
    )

    valid_df[new] = encoder.transform(
        valid_df[original]
    )

    encoders[new] = encoder


with open(
    "label_encoders.pkl",
    "wb"
) as f:

    pickle.dump(
        encoders,
        f
    )


NUM_LEVEL1 = len(encoders["l1"].classes_)

NUM_LEVEL2 = len(encoders["l2"].classes_)

NUM_LEVEL3 = len(encoders["l3"].classes_)

NUM_LEVEL4 = len(encoders["l4"].classes_)

NUM_LEVEL5 = len(encoders["l5"].classes_)

print()

print(NUM_LEVEL1)

print(NUM_LEVEL2)

print(NUM_LEVEL3)

print(NUM_LEVEL4)

print(NUM_LEVEL5)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


class HateSpeechDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self,index):

        row = self.df.iloc[index]

        text = str(row["Input_Text"])

        encoding = tokenizer(

            text,

            max_length=MAX_LEN,

            truncation=True,

            padding="max_length",

            return_tensors="pt"

        )

        return {

            "input_ids":
            encoding["input_ids"].squeeze(),

            "attention_mask":
            encoding["attention_mask"].squeeze(),

            "l1":
            torch.tensor(
                row["l1"],
                dtype=torch.long
            ),

            "l2":
            torch.tensor(
                row["l2"],
                dtype=torch.long
            ),

            "l3":
            torch.tensor(
                row["l3"],
                dtype=torch.long
            ),

            "l4":
            torch.tensor(
                row["l4"],
                dtype=torch.long
            ),

            "l5":
            torch.tensor(
                row["l5"],
                dtype=torch.long
            ),

            "text":
            text

        }

train_dataset = HateSpeechDataset(
    train_df
)

valid_dataset = HateSpeechDataset(
    valid_df
)

train_loader = DataLoader(

    train_dataset,

    batch_size=TRAIN_BATCH_SIZE,

    shuffle=True,

    pin_memory=True

)

valid_loader = DataLoader(

    valid_dataset,

    batch_size=VALID_BATCH_SIZE,

    shuffle=False,

    pin_memory=True

)

print()

print("Training Samples :", len(train_dataset))

print("Validation Samples :", len(valid_dataset))

print()

class HierarchicalClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME
        )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(DROPOUT)

        # Level 1
        self.fc1 = nn.Linear(
            hidden_size,
            NUM_LEVEL1
        )

        # Level 2 receives CLS + Level1 logits
        self.fc2 = nn.Linear(
            hidden_size + NUM_LEVEL1,
            NUM_LEVEL2
        )

        # Level 3 receives CLS + Level2 logits
        self.fc3 = nn.Linear(
            hidden_size + NUM_LEVEL2,
            NUM_LEVEL3
        )

        # Level 4 receives CLS + Level3 logits
        self.fc4 = nn.Linear(
            hidden_size + NUM_LEVEL3,
            NUM_LEVEL4
        )

        # Level 5 receives CLS + Level4 logits
        self.fc5 = nn.Linear(
            hidden_size + NUM_LEVEL4,
            NUM_LEVEL5
        )

    ##########################################################

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        output = self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask

        )

        cls = output.last_hidden_state[:,0]

        cls = self.dropout(cls)

        ######################################################
        # Level 1
        ######################################################

        out1 = self.fc1(cls)

        ######################################################
        # Level 2
        ######################################################

        x2 = torch.cat(
            [cls,out1],
            dim=1
        )

        out2 = self.fc2(x2)

        ######################################################
        # Level 3
        ######################################################

        x3 = torch.cat(
            [cls,out2],
            dim=1
        )

        out3 = self.fc3(x3)

        ######################################################
        # Level 4
        ######################################################

        x4 = torch.cat(
            [cls,out3],
            dim=1
        )

        out4 = self.fc4(x4)

        ######################################################
        # Level 5
        ######################################################

        x5 = torch.cat(
            [cls,out4],
            dim=1
        )

        out5 = self.fc5(x5)

        return {

            "l1":out1,

            "l2":out2,

            "l3":out3,

            "l4":out4,

            "l5":out5

        }

##############################################################
# Model
##############################################################

model = HierarchicalClassifier()

model = model.to(DEVICE)

print(model)

##############################################################
# Class Weights
##############################################################

def build_weights(column):

    weights = compute_class_weight(

        class_weight="balanced",

        classes=np.unique(train_df[column]),

        y=train_df[column]

    )

    return torch.tensor(

        weights,

        dtype=torch.float

    ).to(DEVICE)

##############################################################

weights_l1 = build_weights("l1")

weights_l2 = build_weights("l2")

weights_l3 = build_weights("l3")

weights_l4 = build_weights("l4")

weights_l5 = build_weights("l5")

##############################################################
# Loss Functions
##############################################################

criterion1 = nn.CrossEntropyLoss(
    weight=weights_l1
)

criterion2 = nn.CrossEntropyLoss(
    weight=weights_l2
)

criterion3 = nn.CrossEntropyLoss(
    weight=weights_l3
)

criterion4 = nn.CrossEntropyLoss(
    weight=weights_l4
)

criterion5 = nn.CrossEntropyLoss(
    weight=weights_l5
)

##############################################################
# Optimizer
##############################################################

optimizer = AdamW(

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

scaler = torch.cuda.amp.GradScaler(
    enabled=torch.cuda.is_available()
)

##############################################################
# Early Stopping
##############################################################

best_macro = 0

patience = 3

counter = 0

##############################################################

print()

print("Model Ready")

print()
##############################################################
# PART 3A
# Training and Validation Functions
##############################################################

def calculate_metrics(y_true, y_pred):

    acc = accuracy_score(
        y_true,
        y_pred
    )

    macro = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    weighted = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    return acc, macro, weighted


##############################################################
# Training Function
##############################################################

def train_epoch():

    model.train()

    total_loss = 0

    progress = tqdm(
        train_loader,
        desc="Training"
    )

    for batch in progress:

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        l1 = batch["l1"].to(DEVICE)
        l2 = batch["l2"].to(DEVICE)
        l3 = batch["l3"].to(DEVICE)
        l4 = batch["l4"].to(DEVICE)
        l5 = batch["l5"].to(DEVICE)

        with torch.cuda.amp.autocast():

            outputs = model(

                input_ids,

                attention_mask

            )

            loss1 = criterion1(
                outputs["l1"],
                l1
            )

            loss2 = criterion2(
                outputs["l2"],
                l2
            )

            loss3 = criterion3(
                outputs["l3"],
                l3
            )

            loss4 = criterion4(
                outputs["l4"],
                l4
            )

            loss5 = criterion5(
                outputs["l5"],
                l5
            )

            ##################################################
            # Weighted Hierarchical Loss
            ##################################################

            loss = (

                loss1

                + 1.2 * loss2

                + 1.0 * loss3

                + 1.8 * loss4

                + 1.0 * loss5

            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        scheduler.step()

        total_loss += loss.item()

        progress.set_postfix(

            Loss=round(
                loss.item(),
                4
            )

        )

    return total_loss / len(train_loader)


##############################################################
# Validation Function
##############################################################

def validate():

    model.eval()

    pred = {

        "l1":[],

        "l2":[],

        "l3":[],

        "l4":[],

        "l5":[]

    }

    true = {

        "l1":[],

        "l2":[],

        "l3":[],

        "l4":[],

        "l5":[]

    }

    error_rows = []

    with torch.no_grad():

        for batch in tqdm(

            valid_loader,

            desc="Validation"

        ):

            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(

                input_ids,

                attention_mask

            )

            ##################################################
            # Predictions
            ##################################################

            for level in pred.keys():

                prediction = torch.argmax(

                    outputs[level],

                    dim=1

                )

                pred[level].extend(

                    prediction.cpu().numpy()

                )

                true[level].extend(

                    batch[level].numpy()

                )

            ##################################################
            # Error Analysis
            ##################################################

            p4 = torch.argmax(

                outputs["l4"],

                dim=1

            ).cpu().numpy()

            gt4 = batch["l4"].numpy()

            texts = batch["text"]

            for i in range(len(gt4)):

                if p4[i] != gt4[i]:

                    error_rows.append({

                        "Text":texts[i],

                        "True":gt4[i],

                        "Predicted":p4[i]

                    })

    return pred, true, error_rows

##############################################################
# PART 3B
# Complete Training Pipeline
##############################################################

best_level4_macro = 0.0

patience_counter = 0

training_history = []

##############################################################
# Training Starts
##############################################################

for epoch in range(EPOCHS):

    print("\n" + "="*70)

    print(f"Epoch {epoch+1}/{EPOCHS}")

    print("="*70)

    ##########################################################
    # Train
    ##########################################################

    train_loss = train_epoch()

    ##########################################################
    # Validation
    ##########################################################

    predictions, labels, errors = validate()

    epoch_result = {}

    print("\nValidation Metrics")

    print("-"*70)

    ##########################################################
    # Level Wise Evaluation
    ##########################################################

    for level in ["l1","l2","l3","l4","l5"]:

        acc, macro, weighted = calculate_metrics(

            labels[level],

            predictions[level]

        )

        epoch_result[level] = {

            "accuracy":acc,

            "macro":macro,

            "weighted":weighted

        }

        print(

            f"{level.upper()}"

            f"\tAccuracy : {acc:.4f}"

            f"\tMacro F1 : {macro:.4f}"

            f"\tWeighted F1 : {weighted:.4f}"

        )

    ##########################################################
    # Save Results
    ##########################################################

    training_history.append({

        "epoch":epoch+1,

        "loss":train_loss,

        "l1_macro":epoch_result["l1"]["macro"],

        "l2_macro":epoch_result["l2"]["macro"],

        "l3_macro":epoch_result["l3"]["macro"],

        "l4_macro":epoch_result["l4"]["macro"],

        "l5_macro":epoch_result["l5"]["macro"]

    })

    ##########################################################
    # Save Best Model
    ##########################################################

    current_macro = epoch_result["l4"]["macro"]

    if current_macro > best_level4_macro:

        best_level4_macro = current_macro

        patience_counter = 0

        torch.save(

            {

                "model":model.state_dict(),

                "optimizer":optimizer.state_dict(),

                "epoch":epoch,

                "macro":current_macro

            },

            "best_subtask1_model.pt"

        )

        ######################################################
        # Save Error Analysis
        ######################################################

        pd.DataFrame(errors).to_csv(

            "errors.csv",

            index=False

        )

        print("\nBest model updated.")

        print("Level4 Macro F1 :", round(current_macro,4))

    else:

        patience_counter += 1

        print(

            "\nNo Improvement"

        )

        print(

            "Patience :", patience_counter,

            "/", patience

        )

    ##########################################################
    # Early Stopping
    ##########################################################

    if patience_counter >= patience:

        print("\nEarly Stopping Activated")

        break

##############################################################
# Save Training History
##############################################################

history = pd.DataFrame(training_history)

history.to_csv(

    "training_metrics.csv",

    index=False

)

print()

print("Training Completed Successfully")

print()

print("Best Level4 Macro F1 :", best_level4_macro)