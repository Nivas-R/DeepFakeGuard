import pandas as pd
import os

# ---------- PATHS ----------
EMAIL_REAL = "data/clean/email/real.csv"
EMAIL_FAKE = "data/clean/email/fake.csv"

TEXT_REAL = "data/clean/text/real.csv"
TEXT_FAKE = "data/clean/text/fake.csv"

OUT_DIR = "data/clean/all"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- HELPER ----------
def load_text(csv_path):
    df = pd.read_csv(csv_path)

    # auto-detect text column
    for col in ["text", "Body", "content", "message"]:
        if col in df.columns:
            return df[col].dropna().astype(str).tolist()

    raise ValueError(f"No text column found in {csv_path}")

# ---------- LOAD ----------
print("🔹 Loading cleaned EMAIL data...")
email_real = load_text(EMAIL_REAL)
email_fake = load_text(EMAIL_FAKE)

print("🔹 Loading cleaned TEXT data...")
text_real = load_text(TEXT_REAL)
text_fake = load_text(TEXT_FAKE)

# ---------- MERGE ----------
real_all = email_real + text_real
fake_all = email_fake + text_fake

# ---------- BALANCE ----------
min_len = min(len(real_all), len(fake_all))
real_all = real_all[:min_len]
fake_all = fake_all[:min_len]

print(f"✅ Balanced count → REAL: {len(real_all)} | FAKE: {len(fake_all)}")

# ---------- SAVE ----------
pd.DataFrame({"text": real_all}).to_csv(
    f"{OUT_DIR}/real.csv", index=False
)

pd.DataFrame({"text": fake_all}).to_csv(
    f"{OUT_DIR}/fake.csv", index=False
)

print("🎉 MERGE COMPLETE")
print("Saved to: data/clean/all/")
