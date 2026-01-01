import torch
import re
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ================= CONFIG =================
MODEL_PATH = "./model"
MAX_LEN = 512

LABELS = {
    0: "REAL",
    1: "FAKE"
}

_model = None
_tokenizer = None


# ================= LOAD MODEL ONCE =================
def load_model_once():
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        _model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()  # inference mode
        print("✅ Text model loaded successfully")


# ================= CLEAN TEXT =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ================= MAIN PREDICT FUNCTION =================
def predict_text(text):
    load_model_once()

    # Step 1: clean input
    text = clean_text(text)

    # Step 2: tokenize
    inputs = _tokenizer(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # Step 3: model inference
    with torch.no_grad():
        outputs = _model(**inputs)

    # Step 4: convert logits → probability
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    fake_prob = probs[0][1].item()
    real_prob = probs[0][0].item()

    # Step 5: final decision
    if fake_prob >= 0.5:
        return "FAKE", round(fake_prob, 4)
    else:
        return "REAL", round(real_prob, 4)
