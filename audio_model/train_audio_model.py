# ==========================================
# Professional Audio Deepfake Training Script
# ==========================================

import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -----------------------------
# Configuration
# -----------------------------
DATA_FILE = "audio_mfcc_features.csv"
MODEL_SAVE_PATH = "ml_models/audio_model.pt"
SCALER_SAVE_PATH = "ml_models/scaler.pkl"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 5
SEED = 42

os.makedirs("ml_models", exist_ok=True)

# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Dataset Class
# -----------------------------
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# Load Data
# -----------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["label"]).values
y = df["label"].values

# Train/Val/Test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print("Dataset split complete")

# -----------------------------
# Feature Scaling
# -----------------------------
print("Applying StandardScaler...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"Scaler saved at {SCALER_SAVE_PATH}")

# -----------------------------
# DataLoaders
# -----------------------------
train_loader = DataLoader(AudioDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(AudioDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(AudioDataset(X_test, y_test), batch_size=BATCH_SIZE)

# -----------------------------
# Model Definition
# -----------------------------
class AudioClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AudioClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)

model = AudioClassifier(X.shape[1]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Training Loop with Early Stopping
# -----------------------------
print("\nStarting training...\n")

best_val_loss = float("inf")
counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_preds, train_labels = [], []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        train_labels.extend(y_batch.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)

    # Validation
    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()
            val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        counter = 0
        print("✅ Best model saved")
    else:
        counter += 1
        if counter >= PATIENCE:
            print("⛔ Early stopping triggered")
            break

# -----------------------------
# Final Test Evaluation
# -----------------------------
print("\nEvaluating on test set...")

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

test_preds, test_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        test_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        test_labels.extend(y_batch.cpu().numpy())

print("\nTest Results:")
print("Accuracy :", accuracy_score(test_labels, test_preds))
print("Precision:", precision_score(test_labels, test_preds, zero_division=0))
print("Recall   :", recall_score(test_labels, test_preds, zero_division=0))

print("\n🎉 Training Complete!")
print(f"Model saved at: {MODEL_SAVE_PATH}")
print(f"Scaler saved at: {SCALER_SAVE_PATH}")
