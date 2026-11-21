"""CNN architecture based on the uploaded experimental implementations."""

from __future__ import annotations


def build_model():
    """Build a 10-class CNN for 28x28 grayscale handwritten digit images.

    Pixel scaling is included in the model so training and inference use the
    same preprocessing. Inputs should therefore contain raw pixel values 0-255.
    """
    import tensorflow as tf

    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1), name="grayscale_digit"),
            layers.Rescaling(1.0 / 255.0, name="pixel_rescaling"),
            # Mild geometry changes only; never flip digits horizontally or vertically.
            layers.RandomRotation(0.05, fill_mode="nearest", name="random_rotation"),
            layers.RandomZoom(0.1, fill_mode="nearest", name="random_zoom"),
            layers.Conv2D(32, kernel_size=3, activation="relu", name="conv_32"),
            layers.BatchNormalization(name="bn_32"),
            layers.Conv2D(64, kernel_size=3, activation="relu", name="conv_64"),
            layers.BatchNormalization(name="bn_64"),
            layers.MaxPooling2D(pool_size=2, name="pool_1"),
            layers.Dropout(0.25, name="dropout_1"),
            layers.Conv2D(128, kernel_size=3, activation="relu", name="conv_128"),
            layers.BatchNormalization(name="bn_128"),
            layers.Conv2D(256, kernel_size=3, activation="relu", name="conv_256"),
            layers.BatchNormalization(name="bn_256"),
            layers.MaxPooling2D(pool_size=2, name="pool_2"),
            layers.Dropout(0.25, name="dropout_2"),
            layers.Flatten(name="flatten"),
            layers.Dense(512, activation="relu", name="dense_512"),
            layers.BatchNormalization(name="bn_dense"),
            layers.Dropout(0.5, name="dropout_dense"),
            layers.Dense(10, activation="softmax", name="digit_probabilities"),
        ],
        name="bengali_digit_cnn",
    )
    return model
