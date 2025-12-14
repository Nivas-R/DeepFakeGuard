"""
Test script for DeepFake Image Detector
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepfake_detector import DeepFakeDetector

def test_model():
    print("=" * 60)
    print("🔍 DeepFake Image Detector - Test Script")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading model...")
    detector = DeepFakeDetector()
    
    # Get test image
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = input("\nEnter path to test image: ")
    
    # Check if file exists
    if not os.path.exists(test_image):
        print(f"❌ Error: File not found: {test_image}")
        return
    
    # Predict
    print(f"\n🔍 Analyzing: {test_image}")
    result = detector.predict(test_image)
    
    # Display result
    print("\n" + "=" * 60)
    print("📊 DETECTION RESULT")
    print("=" * 60)
    
    emoji = "❌" if result['is_fake'] else "✅"
    
    print(f"""
    Image: {os.path.basename(test_image)}
    
    {emoji} Result: {result['result']}
    
    Confidence: {result['confidence']:.2f}%
    Raw Score: {result['raw_score']}
    Threshold: 0.65
    """)
    
    # Confidence bar
    confidence = result['confidence']
    filled = int(confidence / 5)
    bar_char = "🟥" if result['is_fake'] else "🟩"
    bar = bar_char * filled + "⬜" * (20 - filled)
    
    print(f"    [{bar}] {confidence:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    test_model()