from __future__ import annotations


def build_model(num_classes: int, image_size: int = 256):
    """Build a compact CNN for experimental four-class chest X-ray classification."""
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(image_size, image_size, 3), name="chest_xray")
    x = inputs
    # Small geometric augmentation only; no horizontal/vertical flips by default.
    x = layers.RandomRotation(0.02, fill_mode="nearest", name="small_rotation")(x)
    x = layers.RandomZoom(0.04, fill_mode="nearest", name="small_zoom")(x)

    for filters, dropout in ((32, 0.10), (64, 0.15), (128, 0.20), (256, 0.25)):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="covid_xray_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
