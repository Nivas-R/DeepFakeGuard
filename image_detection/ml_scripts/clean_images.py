from pathlib import Path
from PIL import Image
import hashlib, os
from tqdm import tqdm

dataset_root = Path("C:/DeepFakeGuard-ML/dataset/image_face")

def is_corrupted(fp):
    try:
        Image.open(fp).verify()
        return False
    except Exception:
        return True

def md5(fp):
    with open(fp,'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

deleted_corrupted = 0
deleted_duplicates = 0
seen = set()
files = [f for f in dataset_root.rglob("*") if f.suffix.lower() in [".jpg",".jpeg",".png"]]
total = len(files)
print(f"🧹 Starting cleaning process for {total:,} images...\n")

for i, file in enumerate(tqdm(files, desc="Cleaning Progress", unit="img")):
    try:
        if is_corrupted(file):
            os.remove(file)
            deleted_corrupted += 1
            continue
        h = md5(file)
        if h in seen:
            os.remove(file)
            deleted_duplicates += 1
        else:
            seen.add(h)
    except Exception:
        continue
    # print progress every 500 files
    if i % 500 == 0 and i > 0:
        print(f"→ Processed {i:,}/{total:,} | "
              f"Corrupted removed: {deleted_corrupted} | "
              f"Duplicates removed: {deleted_duplicates}")

print("\n✅ Cleaning complete!")
print(f"🗑️ Corrupted removed: {deleted_corrupted}")
print(f"📄 Duplicates removed: {deleted_duplicates}")
print(f"📦 Total processed: {total}")
