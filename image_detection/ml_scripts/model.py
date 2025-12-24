import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3)):
    # Load EfficientNetB0 WITHOUT top layer
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Freeze base model (for faster training)
    base_model.trainable = False

    # Add custom layers
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer (REAL = 0, FAKE = 1)
    output = layers.Dense(1, activation='sigmoid')(x)

    # Build final model
    model = models.Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
