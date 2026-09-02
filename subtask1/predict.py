##############################################################
# DravidianCodeMix 2026
# SubTask 1 Prediction Script
##############################################################

import pickle
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel

##############################################################
# Configuration
##############################################################

MODEL_NAME = "FacebookAI/xlm-roberta-large"

MODEL_PATH = "best_subtask1_model.pt"

ENCODER_PATH = "label_encoders.pkl"

TEST_PATH = "../data/subtask1/test.csv"

MAX_LEN = 256

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

##############################################################
# Load Encoders
##############################################################

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

NUM_LEVEL1 = len(encoders["l1"].classes_)
NUM_LEVEL2 = len(encoders["l2"].classes_)
NUM_LEVEL3 = len(encoders["l3"].classes_)
NUM_LEVEL4 = len(encoders["l4"].classes_)
NUM_LEVEL5 = len(encoders["l5"].classes_)

##############################################################
# Tokenizer
##############################################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

##############################################################
# Model
##############################################################

class HierarchicalClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(hidden, NUM_LEVEL1)

        self.fc2 = nn.Linear(hidden + NUM_LEVEL1, NUM_LEVEL2)

        self.fc3 = nn.Linear(hidden + NUM_LEVEL2, NUM_LEVEL3)

        self.fc4 = nn.Linear(hidden + NUM_LEVEL3, NUM_LEVEL4)

        self.fc5 = nn.Linear(hidden + NUM_LEVEL4, NUM_LEVEL5)

    def forward(self, input_ids, attention_mask):

        out = self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask

        )

        cls = self.dropout(out.last_hidden_state[:, 0])

        o1 = self.fc1(cls)

        o2 = self.fc2(torch.cat([cls, o1], dim=1))

        o3 = self.fc3(torch.cat([cls, o2], dim=1))

        o4 = self.fc4(torch.cat([cls, o3], dim=1))

        o5 = self.fc5(torch.cat([cls, o4], dim=1))

        return {
            "l1": o1,
            "l2": o2,
            "l3": o3,
            "l4": o4,
            "l5": o5
        }

##############################################################
# Dataset
##############################################################

class TestDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        encoding = tokenizer(

            str(row["Input_Text"]),

            max_length=MAX_LEN,

            padding="max_length",

            truncation=True,

            return_tensors="pt"

        )

        return {

            "id": row["Input_ID"],

            "input_ids":
                encoding["input_ids"].squeeze(),

            "attention_mask":
                encoding["attention_mask"].squeeze()

        }

##############################################################
# Load Model
##############################################################

model = HierarchicalClassifier()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=torch.device('cpu')
)

model.load_state_dict(
    checkpoint["model"]
)

model.to(DEVICE)

model.eval()

##############################################################
# Test Loader
##############################################################

test_df = pd.read_csv(TEST_PATH)

test_loader = DataLoader(

    TestDataset(test_df),

    batch_size=BATCH_SIZE,

    shuffle=False

)

##############################################################
# Prediction
##############################################################

results = []

with torch.no_grad():

    for batch in test_loader:

        ids = batch["input_ids"].to(DEVICE)

        mask = batch["attention_mask"].to(DEVICE)

        outputs = model(ids, mask)

        p1 = torch.argmax(outputs["l1"], dim=1).cpu().numpy()
        p2 = torch.argmax(outputs["l2"], dim=1).cpu().numpy()
        p3 = torch.argmax(outputs["l3"], dim=1).cpu().numpy()
        p4 = torch.argmax(outputs["l4"], dim=1).cpu().numpy()
        p5 = torch.argmax(outputs["l5"], dim=1).cpu().numpy()

        sample_ids = batch["id"]

        for i in range(len(sample_ids)):

            results.append({

                "Input_ID":
                    int(sample_ids[i]),

                "Level1_Class":
                    encoders["l1"].inverse_transform([p1[i]])[0],

                "Level2_Target":
                    encoders["l2"].inverse_transform([p2[i]])[0],

                "Level3_Category":
                    encoders["l3"].inverse_transform([p3[i]])[0],

                "Level4_Subcategory":
                    encoders["l4"].inverse_transform([p4[i]])[0],

                "Level5_Bias":
                    encoders["l5"].inverse_transform([p5[i]])[0]

            })

##############################################################
# Save Submission
##############################################################

submission = pd.DataFrame(results)

submission.to_csv(

    "submission.csv",

    index=False

)

print()

print("Submission Generated Successfully")

print(submission.head())