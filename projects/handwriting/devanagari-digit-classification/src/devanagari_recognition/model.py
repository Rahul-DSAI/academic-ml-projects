from tensorflow.keras import Sequential, layers


def build_model(num_classes: int = 10, image_size: int = 32, dropout: float = 0.2):
    """Build a CNN based on the uploaded experiment's convolutional structure."""
    augmentation = Sequential(
        [
            layers.RandomRotation(0.025),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomZoom(0.08),
        ],
        name="safe_augmentation",
    )
    model = Sequential(
        [
            layers.Input(shape=(image_size, image_size, 3)),
            augmentation,
            layers.Conv2D(32, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="devanagari_digit_cnn",
    )
    return model
