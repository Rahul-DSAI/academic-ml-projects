import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Predict one Urdu character image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-names", required=True)
    parser.add_argument("--image-size", type=int, default=28)
    args = parser.parse_args()

    class_names = json.loads(Path(args.class_names).read_text(encoding="utf-8"))
    image = Image.open(args.image).convert("RGB").resize((args.image_size, args.image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    probabilities = tf.keras.models.load_model(args.model).predict(array[None, ...], verbose=0)[0]
    index = int(np.argmax(probabilities))
    print(json.dumps({
        "predicted_class": class_names[index],
        "confidence": float(probabilities[index]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
