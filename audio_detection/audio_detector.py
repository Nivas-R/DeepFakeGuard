import joblib
import librosa
import numpy as np

MODEL_PATH = "ml_models/audio_deepfake_model.pkl"

def predict_audio(audio_path):
    model = joblib.load(MODEL_PATH)
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    prediction = model.predict([mfcc_mean])
    return "Fake" if prediction[0] == 1 else "Real"
