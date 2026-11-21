"""Evaluate the best saved checkpoint on a freshly created held-out split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import DEFAULT_CATEGORIES, load_domain_reviews, make_splits
from .metrics import compute_metrics, save_metrics
from .modeling import DynamicPaddingCollator, ReviewDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-dir", default="artifacts/bert-multidomain/best_model")
    parser.add_argument("--output", default="artifacts/bert-multidomain/re-evaluation.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    texts, labels, categories = load_domain_reviews(args.data_root, DEFAULT_CATEGORIES)
    splits = make_splits(texts, labels, categories, args.test_size, args.val_size, args.seed)
    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(ReviewDataset(splits.test_texts, splits.test_labels),
                        batch_size=args.batch_size, shuffle=False,
                        collate_fn=DynamicPaddingCollator(tokenizer, args.max_length))
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels_batch = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            y_true.extend(labels_batch.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, categories)
    save_metrics(metrics, args.output)
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                      "weighted_f1": metrics["weighted_f1"]}, indent=2))


if __name__ == "__main__":
    main()
