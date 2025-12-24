import torch
import re
import pdfplumber
import email
from email import policy
from docx import Document
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ================= CONFIG =================
MODEL_PATH = r"C:\DeepFakeGuard-Text detector\text_email_detection\model"
MAX_LEN = 512

LABELS = {
    0: "REAL / SAFE",
    1: "SCAM / FAKE"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TEXT CLEANING =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================= RULE-BASED CHECK (STRONG ONLY) =================
def rule_based_scam_check(text):
    text = text.lower()

    strong_patterns = [
        "lottery", "won prize", "claim now", "free money",
        "guaranteed profit", "registration fee",
        "crypto investment", "earn daily"
    ]

    if any(p in text for p in strong_patterns):
        return True

    if "otp" in text and any(w in text for w in ["click", "verify", "link", "urgent"]):
        return True

    return False

# ================= LOAD MODEL =================
tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_PATH, local_files_only=True
)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH, local_files_only=True
)

model.to(device)
model.eval()

# ================= CORE PREDICT =================
def predict(text):
    cleaned = clean_text(text)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100

    label = LABELS[pred]

    # Hybrid safety override
    if rule_based_scam_check(text) and confidence < 90:
        return "SCAM / FAKE (Hybrid Detection)", round(max(confidence, 90), 2)

    return label, round(confidence, 2)

# ================= FILE EXTRACTORS =================
def extract_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def extract_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_eml(path):
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_content()
    else:
        return msg.get_content()

    return ""

# ================= MAIN =================
if __name__ == "__main__":
    print("\n🔍 DeepFakeGuard – Any Text / PDF / Word / Email Detection")
    print("Choose input type:")
    print("1 - Plain Text")
    print("2 - PDF File")
    print("3 - Word File (.docx)")
    print("4 - Email File (.eml)")

    choice = input("Enter choice (1/2/3/4): ").strip()

    if choice == "1":
        print("\nPaste text (CTRL+Z then ENTER):\n")
        text = ""
        while True:
            try:
                text += input() + "\n"
            except EOFError:
                break

    elif choice == "2":
        path = input("Enter PDF file path: ")
        text = extract_pdf(path)

    elif choice == "3":
        path = input("Enter DOCX file path: ")
        text = extract_docx(path)

    elif choice == "4":
        path = input("Enter EML file path: ")
        text = extract_eml(path)

    else:
        print("❌ Invalid option")
        exit()

    label, confidence = predict(text)

    print("\n================ RESULT ================")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence}%")
    print("=======================================\n")
