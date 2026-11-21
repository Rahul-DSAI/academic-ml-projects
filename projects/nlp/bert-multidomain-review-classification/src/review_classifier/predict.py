from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict(text: str, model_dir: str | Path, max_length: int = 256):
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    encoded = tokenizer(text, return_tensors="pt", truncation=True,
                        padding=True, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=1)[0]
    class_id = int(probabilities.argmax().item())
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    return {
        "predicted_domain": id2label.get(class_id, str(class_id)),
        "class_id": class_id,
        "confidence": float(probabilities[class_id].item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict the product domain of a review.")
    parser.add_argument("text", help="Review text to classify")
    parser.add_argument("--model-dir", default="artifacts/bert-multidomain/best_model")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(predict(args.text, args.model_dir, args.max_length), indent=2))


if __name__ == "__main__":
    main()
