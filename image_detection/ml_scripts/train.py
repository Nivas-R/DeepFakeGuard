import os
import tensorflow as tf
from preprocess import load_dataset
from model import build_model

# Paths
BASE_PATH = r"C:\DeepFakeGuard-ML\dataset_split"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
VAL_PATH = os.path.join(BASE_PATH, "val")
TEST_PATH = os.path.join(BASE_PATH, "test")

MODEL_SAVE_PATH = r"C:\DeepFakeGuard-ML\ml_models\deepfake_model.h5"

# Load Datasets
print("📥 Loading datasets...")
train_ds = load_dataset(TRAIN_PATH)
val_ds = load_dataset(VAL_PATH)
test_ds = load_dataset(TEST_PATH)

# Build Model
print("🔧 Building model...")
model = build_model()

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    patience=5,
    monitor='val_loss',
    restore_best_weights=True
)

# Train Model
print("🚀 Training started...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[checkpoint, early_stop]
)

print("✅ Training complete!")
print(f"Model saved to: {MODEL_SAVE_PATH}")

# Evaluate Model
print("📊 Evaluating model on test dataset...")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
