import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ---------------- CONFIG ----------------
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 8
EPOCHS = 3
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD DATA ----------------
def load_data(real_path, fake_path):
    real = pd.read_csv(real_path)
    fake = pd.read_csv(fake_path)

    real["label"] = 0
    fake["label"] = 1

    df = pd.concat([real, fake]).sample(frac=1).reset_index(drop=True)
    return df["text"].astype(str).tolist(), df["label"].tolist()

train_texts, train_labels = load_data(
    "dataset_split/train/real.csv",
    "dataset_split/train/fake.csv"
)

val_texts, val_labels = load_data(
    "dataset_split/val/real.csv",
    "dataset_split/val/fake.csv"
)

# ---------------- DATASET ----------------
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_ds = TextDataset(train_texts, train_labels)
val_ds = TextDataset(val_texts, val_labels)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ---------------- MODEL ----------------
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-5)

# ---------------- TRAIN ----------------
print("🚀 Training started...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    # ---------- VALIDATION ----------
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {total_loss:.4f}")
    print(f"Val Accuracy: {acc:.4f}")
    print(f"Val F1-score: {f1:.4f}")

# ---------------- SAVE MODEL ----------------
model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("\n✅ MODEL TRAINED & SAVED in /model folder")
