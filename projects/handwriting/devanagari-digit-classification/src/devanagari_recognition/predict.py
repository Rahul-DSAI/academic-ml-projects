from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from .data import image_for_model


def main():
    p = argparse.ArgumentParser(description="Predict one Devanagari numeral image")
    p.add_argument("image"); p.add_argument("--model", default="models/best.keras")
    p.add_argument("--classes", default="models/classes.json"); p.add_argument("--image-size", type=int, default=32)
    a = p.parse_args()
    image = Image.open(a.image).convert("RGB")
    # Invert only is not done automatically: keep original foreground/background polarity.
    arr = image_for_model(np.asarray(image), a.image_size)[None, ...]
    model = tf.keras.models.load_model(a.model)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs)); names = json.loads(Path(a.classes).read_text(encoding="utf-8"))
    print(json.dumps({"predicted_class": names[idx], "class_index": idx,
                       "confidence": float(probs[idx])}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
