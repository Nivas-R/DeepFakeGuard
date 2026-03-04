import sys
from deepfake_detector import DeepFakeDetector

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    exit(1)

image_path = sys.argv[1]

detector = DeepFakeDetector()
label, confidence = detector.predict(image_path)

print(f"{label} – {confidence}%")
