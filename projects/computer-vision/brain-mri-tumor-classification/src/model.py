"""CNN architecture based on the 128x128 model in the supplied MRI notebook."""

import tensorflow as tf
from tensorflow.keras import Sequential, layers


def build_cnn(image_size: int, num_classes: int) -> tf.keras.Model:
    if image_size < 32:
        raise ValueError("image_size must be at least 32 pixels")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    model = Sequential(
        [
            layers.Input(shape=(image_size, image_size, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="brain_mri_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
