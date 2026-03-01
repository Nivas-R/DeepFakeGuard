import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "data/audio"
FEATURE_DIR = "data/audio/features"
CATEGORIES = ["real", "fake"]

SR = 16000
N_MFCC = 40

for category in CATEGORIES:
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(FEATURE_DIR, split, category), exist_ok=True)

for category in CATEGORIES:
    files = [f for f in os.listdir(os.path.join(DATA_DIR, category)) if f.endswith(".wav")]
    
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

    splits = {"train": train_files, "val": val_files, "test": test_files}

    for split, file_list in splits.items():
        for file_name in file_list:
            file_path = os.path.join(DATA_DIR, category, file_name)
            y, sr = librosa.load(file_path, sr=SR, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            mfcc = mfcc.T
            save_path = os.path.join(FEATURE_DIR, split, category, file_name.replace(".wav", ".npy"))
            np.save(save_path, mfcc)

print("Preprocessing completed successfully!")
