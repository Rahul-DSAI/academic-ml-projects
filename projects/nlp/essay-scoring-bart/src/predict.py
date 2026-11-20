"""Predict a 0–3 Score1 class for one response using a fine-tuned checkpoint."""

from __future__ import annotations

import argparse
import json

import torch

from src.model import load_finetuned_model
from src.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="artifacts/best_model")
    parser.add_argument("--text", required=True, help="Student response text to score")
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text.strip():
        raise ValueError("--text must contain a non-empty response.")
    tokenizer, model = load_finetuned_model(args.model_dir)
    device = get_device()
    model.to(device)
    model.eval()

    inputs = tokenizer(
        args.text,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_score = int(torch.argmax(probabilities).item())

    result = {
        "predicted_score1": predicted_score,
        "class_probabilities": {
            str(index): round(float(probability), 6)
            for index, probability in enumerate(probabilities.cpu().tolist())
        },
        "warning": (
            "Experimental model output. This checkpoint predicts Score1 classes 0–3 and "
            "does not condition on EssaySet or a prompt-specific rubric. It must not be "
            "used as the sole basis for real educational grading decisions."
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
