"""Predict a class for one MRI image using a saved model."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import DEFAULT_CONFIG
from src.utils import load_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an MRI image")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_CONFIG.output_dir / "best_model.keras")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_CONFIG.output_dir / "metadata.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_path}. Train first.")

    metadata = load_json(args.metadata_path)
    class_names = metadata["class_names"]
    image_size = int(metadata["image_size"])

    image = tf.keras.utils.load_img(
        args.image, target_size=(image_size, image_size), color_mode="rgb"
    )
    array = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
    batch = np.expand_dims(array, axis=0)

    model = tf.keras.models.load_model(args.model_path)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    print(f"Predicted dataset label: {class_names[predicted_index]}")
    print("Class probabilities:")
    for index in np.argsort(probabilities)[::-1]:
        print(f"  {class_names[index]}: {probabilities[index]:.4f}")
    print("\nResearch/educational output only; not a medical diagnosis.")


if __name__ == "__main__":
    main()
