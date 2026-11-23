from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experimental prediction on one chest X-ray image.")
    parser.add_argument("image", help="Path to a chest X-ray image")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    import tensorflow as tf
    from PIL import Image

    root = Path(args.artifacts_dir)
    config = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
    image_size = int(config["image_size"])
    class_names = config["class_names"]
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
    model = tf.keras.models.load_model(root / "best_model.keras")
    probabilities = model.predict(array[np.newaxis, ...], verbose=0)[0]
    ranking = np.argsort(probabilities)[::-1]
    print("Experimental model output (not a medical diagnosis):")
    for idx in ranking:
        print(f"{class_names[int(idx)]}: {probabilities[idx]:.4f}")


if __name__ == "__main__":
    main()
