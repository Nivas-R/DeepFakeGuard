import os
import pandas as pd
import re
import random

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_clean(folder):
    texts = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            print("Reading:", path)

            try:
                df = pd.read_csv(path)
            except:
                df = pd.read_csv(path, encoding="latin-1")

            # detect text column
            col = None
            for c in df.columns:
                if c.lower() in ["text", "body", "content", "message"]:
                    col = c
                    break
            if col is None:
                col = df.columns[0]

            for t in df[col]:
                t = clean_text(t)
                if len(t.split()) > 5:
                    texts.append(t)

    return texts

def balance_and_save(real, fake, out_dir):
    random.shuffle(real)
    random.shuffle(fake)

    size = min(len(real), len(fake))
    real = real[:size]
    fake = fake[:size]

    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame({"text": real}).to_csv(
        os.path.join(out_dir, "real.csv"), index=False
    )
    pd.DataFrame({"text": fake}).to_csv(
        os.path.join(out_dir, "fake.csv"), index=False
    )

    print(f"✅ Saved {size} REAL and {size} FAKE to {out_dir}")

# ================= EMAIL =================
print("\n🔹 Processing EMAIL data")
email_real = load_and_clean("data/raw/email/real")
email_fake = load_and_clean("data/raw/email/fake")
balance_and_save(email_real, email_fake, "data/clean/email")

# ================= TEXT ==================
print("\n🔹 Processing TEXT data")
text_real = load_and_clean("data/raw/text/real")
text_fake = load_and_clean("data/raw/text/fake")
balance_and_save(text_real, text_fake, "data/clean/text")

print("\n🎉 ALL DATA CLEANED & BALANCED SUCCESSFULLY")
