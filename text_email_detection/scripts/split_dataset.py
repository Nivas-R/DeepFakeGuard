import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ---------- PATHS ----------
REAL_PATH = "data/clean/all/real.csv"
FAKE_PATH = "data/clean/all/fake.csv"

OUT_DIR = "dataset_split"

# ---------- CREATE FOLDERS ----------
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUT_DIR}/{split}", exist_ok=True)

# ---------- LOAD DATA ----------
real_df = pd.read_csv(REAL_PATH)
fake_df = pd.read_csv(FAKE_PATH)

# ---------- SPLIT REAL ----------
real_train, real_temp = train_test_split(
    real_df, test_size=0.2, random_state=42
)

real_val, real_test = train_test_split(
    real_temp, test_size=0.5, random_state=42
)

# ---------- SPLIT FAKE ----------
fake_train, fake_temp = train_test_split(
    fake_df, test_size=0.2, random_state=42
)

fake_val, fake_test = train_test_split(
    fake_temp, test_size=0.5, random_state=42
)

# ---------- SAVE ----------
real_train.to_csv(f"{OUT_DIR}/train/real.csv", index=False)
real_val.to_csv(f"{OUT_DIR}/val/real.csv", index=False)
real_test.to_csv(f"{OUT_DIR}/test/real.csv", index=False)

fake_train.to_csv(f"{OUT_DIR}/train/fake.csv", index=False)
fake_val.to_csv(f"{OUT_DIR}/val/fake.csv", index=False)
fake_test.to_csv(f"{OUT_DIR}/test/fake.csv", index=False)

# ---------- SUMMARY ----------
print("✅ DATASET SPLIT COMPLETED")
print("Train:", len(real_train), "real |", len(fake_train), "fake")
print("Val  :", len(real_val), "real |", len(fake_val), "fake")
print("Test :", len(real_test), "real |", len(fake_test), "fake")
