"""Predict the digit class of a single handwritten digit image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import CLASS_NAMES, IMAGE_SIZE


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """Load an image as a 28x28 grayscale array in the model's raw 0-255 range."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        grayscale = ImageOps.grayscale(image)
        resized = grayscale.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32)
    return array[np.newaxis, ..., np.newaxis]


def main() -> None:
    import tensorflow as tf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an input digit image")
    parser.add_argument("--model", default="artifacts/best_model.keras")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}. Train the model first.")
    model = tf.keras.models.load_model(model_path)
    image = preprocess_image(args.image)
    probabilities = model.predict(image, verbose=0)[0]
    prediction = int(np.argmax(probabilities))

    print(f"Predicted class: {CLASS_NAMES[prediction]}")
    print(f"Confidence: {probabilities[prediction]:.4f}")
    print("Class probabilities:")
    for name, probability in zip(CLASS_NAMES, probabilities):
        print(f"  {name}: {probability:.4f}")


if __name__ == "__main__":
    main()
