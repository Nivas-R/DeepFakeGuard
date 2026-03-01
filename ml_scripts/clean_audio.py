import os
import librosa
import soundfile as sf
import noisereduce as nr

INPUT_ROOT = r"C:/DeepFakeGuard-ML/audio_dataset"
OUTPUT_ROOT = r"C:/DeepFakeGuard-ML/audio_dataset/cleaned_audio"

def clean_audio_file(input_path, output_path):
    try:
        audio, sr = librosa.load(input_path, sr=16000)
        audio, _ = librosa.effects.trim(audio)
        audio = nr.reduce_noise(y=audio, sr=sr)
        audio = librosa.util.normalize(audio)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, sr)

        print(f"Cleaned: {output_path}")

    except Exception as e:
        print(f"Error cleaning {input_path}: {e}")

def clean_folder(split):
    categories = ["real", "fake"]
    for category in categories:
        input_dir = os.path.join(INPUT_ROOT, split, category)
        output_dir = os.path.join(OUTPUT_ROOT, split, category)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".wav", ".mp3", ".flac")):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename.replace(".mp3", ".wav"))
                clean_audio_file(input_path, output_path)

if __name__ == "__main__":
    for split in ["train", "test", "validation"]:
        clean_folder(split)
    print("Cleaning Finished!")
