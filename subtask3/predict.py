##############################################################
# FIRE 2026
# Subtask 3
# Prediction Script
##############################################################

import pickle
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

##############################################################
# Configuration
##############################################################

MODEL_NAME = "FacebookAI/xlm-roberta-base"

MODEL_PATH = "best_subtask3_model.pt"

LABEL_ENCODER = "label_encoder.pkl"

TEST_PATH = "../data/subtask3/test.csv"

MAX_LEN = 128

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

##############################################################
# Load Label Encoder
##############################################################

with open(LABEL_ENCODER, "rb") as f:
    label_encoder = pickle.load(f)

NUM_CLASSES = len(label_encoder.classes_)

##############################################################
# Tokenizer
##############################################################

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
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

    def forward(self,input_ids,attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()

        pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        pooled = self.dropout(pooled)

        return self.classifier(pooled)

##############################################################
# Test Dataset
##############################################################

class TestDataset(Dataset):

    def __init__(self,df):

        self.df = df.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self,index):

        row = self.df.iloc[index]

        text = str(row["Comments"])

        encoding = tokenizer(

            text,

            truncation=True,

            padding="max_length",

            max_length=MAX_LEN,

            return_tensors="pt"

        )

        return{

            "id":row["ID"],

            "comment":text,

            "input_ids":encoding["input_ids"].squeeze(),

            "attention_mask":encoding["attention_mask"].squeeze()

        }

##############################################################
# Load Model
##############################################################

model = KannadaClassifier().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()

##############################################################
# Load Test
##############################################################

test_df = pd.read_csv(TEST_PATH)

test_df = test_df.drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)

test_loader = DataLoader(

    TestDataset(test_df),

    batch_size=BATCH_SIZE,

    shuffle=False

)

##############################################################
# Prediction
##############################################################

ids = []

comments = []

predictions = []

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        logits = model(

            input_ids,

            attention_mask

        )

        pred = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy()

        ids.extend(batch["id"])

        comments.extend(batch["comment"])

        predictions.extend(pred)

##############################################################
# Decode Labels
##############################################################

labels = label_encoder.inverse_transform(predictions)

##############################################################
# Save Submission
##############################################################

submission = pd.DataFrame({

    "id":ids,

    "label_pred":labels

})

submission.to_csv(
    "Innovators.tsv",
    sep="\t",
    index=False
)

print()

print("Submission Saved Successfully!")

print()

print(submission.head())