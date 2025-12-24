import tensorflow as tf

def load_dataset(data_dir, img_size=(224, 224), batch_size=32):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary"
    )

    # Performance boost
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
