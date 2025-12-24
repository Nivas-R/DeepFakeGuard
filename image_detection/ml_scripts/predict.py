import tensorflow as tf
import argparse
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = r"C:\DeepFakeGuard-ML\ml_models\deepfake_model.h5"

# Load the trained model
print("📥 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Fake >= 0.5 | Real < 0.5
    label = "FAKE" if prediction >= 0.5 else "REAL"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    print("===================================")
    print(f"Image: {img_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("===================================")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    args = parser.parse_args()

    predict_image(args.image)
