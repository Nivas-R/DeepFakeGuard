"""
DeepFake Image Detector
Model: deepfake_detector_FINAL.keras
Threshold: 0.65 (Optimized)
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

class DeepFakeDetector:
    """
    DeepFake Image Detection Model
    Trained on 314,572 images
    Val AUC: 0.8048
    """
    
    def __init__(self, model_path=None, threshold=0.65):
        """
        Initialize detector
        
        Args:
            model_path: Path to .keras model file (optional)
            threshold: Decision threshold (0.65 recommended)
        """
        # Default model path
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'ml_models', 'deepfake_detector_FINAL.keras')
        
        self.model = load_model(model_path)
        self.threshold = threshold
        print(f"✅ DeepFake Detector loaded (Threshold: {threshold})")
    
    def predict(self, image_path):
        """
        Detect if image is real or deepfake
        
        Args:
            image_path: Path to image file (jpg, png)
            
        Returns:
            dict: {
                'is_fake': bool,
                'result': str ('REAL' or 'FAKE'),
                'confidence': float (0-100),
                'raw_score': float (0-1)
            }
        """
        # Load and resize to 224x224
        img = image.load_img(image_path, target_size=(224, 224))
        
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        raw_score = self.model.predict(img_array, verbose=0)[0][0]
        
        # Determine result
        if raw_score > self.threshold:
            is_fake = False
            result = 'REAL'
            confidence = raw_score * 100
        else:
            is_fake = True
            result = 'FAKE'
            confidence = (1 - raw_score) * 100
        
        return {
            'is_fake': is_fake,
            'result': result,
            'confidence': round(confidence, 2),
            'raw_score': round(float(raw_score), 4)
        }
    
    def predict_batch(self, image_paths):
        """
        Predict multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)
        return results


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Initialize detector
    detector = DeepFakeDetector()
    
    # Test image
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        result = detector.predict(test_image)
        
        print("\n" + "=" * 50)
        print("RESULT")
        print("=" * 50)
        print(f"Result: {result['result']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Is Fake: {result['is_fake']}")
        print(f"Raw Score: {result['raw_score']}")
        print("=" * 50)
    else:
        print("Usage: python deepfake_detector.py <image_path>")