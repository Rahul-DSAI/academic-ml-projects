"""Predict a Kannada-MNIST class from a 28x28 grayscale image file."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return array[np.newaxis, ..., np.newaxis]


def predict(image_path, model_path):
    model = tf.keras.models.load_model(model_path)
    batch = preprocess_image(image_path)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a digit image")
    parser.add_argument("--model", default="models/best_model.keras")
    args = parser.parse_args()
    if not Path(args.image).is_file():
        parser.error(f"Image not found: {args.image}")
    predict(args.image, args.model)


if __name__ == "__main__":
    main()
