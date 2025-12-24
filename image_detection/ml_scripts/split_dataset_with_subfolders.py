import random, shutil
from pathlib import Path
from tqdm import tqdm

# Input folder (your cleaned dataset)
root = Path("C:/DeepFakeGuard-ML/dataset/image_face")

# Output folder (new split data will be saved here)
out = Path("C:/DeepFakeGuard-ML/dataset_split")

# Split ratios
ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
random.seed(42)

# Create folder structure
for s in ratios:
    for lbl in ["real", "fake"]:
        (out / s / lbl).mkdir(parents=True, exist_ok=True)

def process_label(label):
    src_label = root / label
    for sub in [p for p in src_label.iterdir() if p.is_dir()]:
        imgs = [p for p in sub.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(ratios["train"] * n)
        n_val = int(ratios["val"] * n)
        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:]
        }
        for s, files in splits.items():
            dest = out / s / label / sub.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in tqdm(files, desc=f"{label}/{sub.name} -> {s}", leave=False):
                shutil.copy2(f, dest / f.name)

process_label("real")
process_label("fake")
print("\n✅ Dataset split complete!")
print("Output saved in: C:/DeepFakeGuard-ML/dataset_split/")
