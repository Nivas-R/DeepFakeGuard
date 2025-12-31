import numpy as np
import tensorflow as tf
from preprocess import preprocess_image

MODEL_PATH = "../ml_models/deepfake_image_model.keras"

class DeepFakeDetector:
    def __init__(self):
        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

    def predict(self, image_path):
        image = preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)

        fake_prob = float(self.model.predict(image)[0][0])

        # FINAL DECISION LOGIC
        if fake_prob >= 0.5:
            label = "FAKE"
            confidence = fake_prob * 100
        else:
            label = "REAL"
            confidence = (1 - fake_prob) * 100

        return label, round(confidence, 2)
