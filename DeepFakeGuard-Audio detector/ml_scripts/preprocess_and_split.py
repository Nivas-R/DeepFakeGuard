import os
import librosa
import soundfile as sf

INPUT_DIR = "audio_dataset/cleaned_audio"
OUTPUT_DIR = "audio_dataset/processed_audio"

SAMPLE_RATE = 16000
DURATION = 3  # seconds

def preprocess_audio(inp, out):
    audio, _ = librosa.load(inp, sr=SAMPLE_RATE)
    audio = audio[:SAMPLE_RATE * DURATION]
    audio = librosa.util.normalize(audio)
    sf.write(out, audio, SAMPLE_RATE)

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        in_dir = os.path.join(INPUT_DIR, split, label)
        out_dir = os.path.join(OUTPUT_DIR, split, label)

        if not os.path.exists(in_dir):
            print(f"⚠️ Skipping missing folder: {in_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)

        for f in os.listdir(in_dir):
            if f.endswith(".wav"):
                preprocess_audio(
                    os.path.join(in_dir, f),
                    os.path.join(out_dir, f)
                )

print("✅ Audio preprocessing saved to processed_audio")

