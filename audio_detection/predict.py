import sys
from audio_detector import predict_audio

if __name__ == "__main__":
    audio_file = sys.argv[1]
    result = predict_audio(audio_file)
    print("Prediction:", result)
