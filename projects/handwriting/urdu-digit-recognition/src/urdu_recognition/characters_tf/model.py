import tensorflow as tf
from tensorflow.keras import layers, models


def build_character_cnn(num_classes: int, image_size: int = 28) -> tf.keras.Model:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    inputs = layers.Input(shape=(image_size, image_size, 3), name="image")
    x = layers.Conv2D(32, 3, activation="relu", padding="valid")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return models.Model(inputs, outputs, name="urdu_character_cnn")
