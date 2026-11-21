from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from .data import DEFAULT_CATEGORIES, load_domain_reviews, make_splits
from .metrics import compute_metrics, save_metrics
from .modeling import DynamicPaddingCollator, ReviewDataset, load_tokenizer_and_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for product-domain review classification.")
    parser.add_argument("--data-root", required=True, help="Path to unprocessed/sorted_data")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--output-dir", default="artifacts/bert-multidomain")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional per-run cap for quick smoke tests; split is made before this cap.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loader(model, loader, device):
    model.eval()
    y_true, y_pred, total_loss, steps = [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            steps += 1
            y_true.extend(batch["labels"].cpu().tolist())
            y_pred.extend(outputs.logits.argmax(dim=1).cpu().tolist())
    return y_true, y_pred, total_loss / max(steps, 1)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        raise ValueError("epochs/batch-size must be positive and max-length must be >= 8")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, labels, categories = load_domain_reviews(args.data_root, DEFAULT_CATEGORIES)
    splits = make_splits(texts, labels, categories, args.test_size, args.val_size, args.seed)
    train_texts = splits.train_texts
    train_labels = splits.train_labels
    if args.max_train_samples:
        train_texts = train_texts[:args.max_train_samples]
        train_labels = train_labels[:args.max_train_samples]

    (output_dir / "label_mapping.json").write_text(
        json.dumps({str(i): name for i, name in enumerate(categories)}, indent=2), encoding="utf-8"
    )
    (output_dir / "split_summary.json").write_text(json.dumps({
        "train": len(train_texts), "validation": len(splits.val_texts),
        "test": len(splits.test_texts), "categories": categories, "seed": args.seed,
    }, indent=2), encoding="utf-8")

    tokenizer, model = load_tokenizer_and_model(args.model_name, categories)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    collator = DynamicPaddingCollator(tokenizer, args.max_length)
    train_loader = DataLoader(ReviewDataset(train_texts, train_labels),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collator)
    val_loader = DataLoader(ReviewDataset(splits.val_texts, splits.val_labels),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collator)
    test_loader = DataLoader(ReviewDataset(splits.test_texts, splits.test_labels),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_val_f1 = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum, seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch_size = batch["labels"].size(0)
            loss_sum += outputs.loss.item() * batch_size
            seen += batch_size
            progress.set_postfix(train_loss=f"{loss_sum / max(seen, 1):.4f}")

        val_true, val_pred, val_loss = evaluate_loader(model, val_loader, device)
        val_metrics = compute_metrics(val_true, val_pred, categories)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(seen, 1),
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
            save_metrics(val_metrics, output_dir / "best_validation_metrics.json")

    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Evaluate the selected checkpoint once on the held-out test split.
    best_model = type(model).from_pretrained(output_dir / "best_model").to(device)
    test_true, test_pred, test_loss = evaluate_loader(best_model, test_loader, device)
    test_metrics = compute_metrics(test_true, test_pred, categories)
    test_metrics["loss"] = test_loss
    save_metrics(test_metrics, output_dir / "test_metrics.json")
    print("Held-out test metrics:", json.dumps({k: v for k, v in test_metrics.items()
          if k not in {"classification_report", "confusion_matrix"}}, indent=2))
    print(f"Saved model and reports to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
