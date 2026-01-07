import librosa
import numpy as np
import pandas as pd
import os

DATASET_PATH = "audio_dataset/processed_audio/train"
OUTPUT_CSV = "audio_mfcc_features.csv"

features = []
labels = []

for label in ["real", "fake"]:
    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            try:
                y, sr = librosa.load(file_path, sr=None)

                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc.T, axis=0)

                features.append(mfcc_mean)
                labels.append(1 if label == "real" else 0)

            except Exception as e:
                print(f"Error processing {file}: {e}")

df = pd.DataFrame(features)
df["label"] = labels

df.to_csv(OUTPUT_CSV, index=False)
print("✅ MFCC feature extraction completed!")
print(f"📁 Saved as: {OUTPUT_CSV}")
