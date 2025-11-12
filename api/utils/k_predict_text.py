# api/utils/k_predict_text.py
def predict_text(text_input):
    """
    Placeholder function for text model prediction.
    Later this will load K's text_model.pt or tokenizer.
    """
    # Dummy prediction logic
    if "fake" in text_input.lower():
        return "Fake", 0.82
    else:
        return "Real", 0.64
