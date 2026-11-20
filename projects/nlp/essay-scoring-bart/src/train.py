"""Fine-tune BART for four-class ASAP-SAS Score1 prediction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import Config
from src.data import DynamicPaddingCollator, EssayScoringDataset, load_data, split_data
from src.engine import evaluate_predictions, optimizer_steps_per_epoch, predict_loader, train_one_epoch
from src.model import ID2LABEL, LABEL2ID, load_model, load_tokenizer
from src.utils import ensure_dir, get_device, make_logger, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_config.yaml", help="YAML config path")
    parser.add_argument("--data", default=None, help="Optional override for the training TSV path")
    parser.add_argument("--output-dir", default=None, help="Optional override for output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    data_path = args.data or config.data_path
    output_dir = Path(args.output_dir or config.output_dir)
    ensure_dir(output_dir)
    logger = make_logger()

    set_seed(config.random_seed)
    device = get_device()
    logger.info("Device: %s", device)
    logger.info("Loading data from %s", data_path)

    frame = load_data(data_path)
    logger.info("Loaded %d usable rows", len(frame))
    logger.info("Score1 distribution: %s", frame["Score1"].value_counts().sort_index().to_dict())
    logger.info("EssaySet distribution: %s", frame["EssaySet"].value_counts().sort_index().to_dict())

    train_frame, validation_frame, test_frame = split_data(
        frame,
        test_size=config.test_size,
        validation_size=config.validation_size,
        random_seed=config.random_seed,
    )
    split_summary = {
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "test_rows": len(test_frame),
        "train_score_counts": train_frame["Score1"].value_counts().sort_index().to_dict(),
        "validation_score_counts": validation_frame["Score1"].value_counts().sort_index().to_dict(),
        "test_score_counts": test_frame["Score1"].value_counts().sort_index().to_dict(),
        "split_method": "row-level random split stratified by Score1",
        "random_seed": config.random_seed,
    }
    save_json(split_summary, output_dir / "split_summary.json")

    tokenizer = load_tokenizer(config.model_name_or_path)
    model = load_model(config.model_name_or_path, num_labels=config.num_labels)
    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL
    model.to(device)

    collator = DynamicPaddingCollator(tokenizer)
    common_loader_args = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collator,
    }
    train_loader = DataLoader(
        EssayScoringDataset(train_frame, tokenizer, config.max_length),
        batch_size=config.train_batch_size,
        shuffle=True,
        **common_loader_args,
    )
    validation_loader = DataLoader(
        EssayScoringDataset(validation_frame, tokenizer, config.max_length),
        batch_size=config.eval_batch_size,
        shuffle=False,
        **common_loader_args,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_optimizer_steps = optimizer_steps_per_epoch(
        len(train_loader), config.gradient_accumulation_steps
    ) * config.num_train_epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    best_qwk = -math.inf
    epochs_without_improvement = 0
    best_model_dir = output_dir / "best_model"
    history: list[dict] = []

    for epoch in range(1, config.num_train_epochs + 1):
        logger.info("Epoch %d/%d", epoch, config.num_train_epochs)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            use_amp=config.use_amp,
        )
        val_results = predict_loader(model, validation_loader, device, use_amp=config.use_amp)
        val_metrics = evaluate_predictions(val_results)
        qwk = val_metrics["quadratic_weighted_kappa"]
        qwk_for_selection = qwk if qwk is not None else -math.inf

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_metrics": val_metrics,
        }
        history.append(epoch_record)
        logger.info(
            "Epoch %d | train loss %.4f | train accuracy %.4f | val accuracy %.4f | val QWK %s",
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["accuracy"],
            f"{qwk:.4f}" if qwk is not None else "undefined",
        )

        if qwk_for_selection > best_qwk:
            best_qwk = qwk_for_selection
            epochs_without_improvement = 0
            ensure_dir(best_model_dir)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            save_json(
                {"epoch": epoch, "validation_metrics": val_metrics},
                output_dir / "best_validation_metrics.json",
            )
            logger.info("Saved new best checkpoint to %s", best_model_dir)
        else:
            epochs_without_improvement += 1

        save_json({"history": history}, output_dir / "training_history.json")
        pd.DataFrame(
            [
                {
                    "epoch": item["epoch"],
                    "train_loss": item["train_loss"],
                    "train_accuracy": item["train_accuracy"],
                    "validation_accuracy": item["validation_metrics"]["accuracy"],
                    "validation_mae": item["validation_metrics"]["mae"],
                    "validation_qwk": item["validation_metrics"]["quadratic_weighted_kappa"],
                }
                for item in history
            ]
        ).to_csv(output_dir / "training_history.csv", index=False)

        if config.early_stopping_patience > 0 and epochs_without_improvement >= config.early_stopping_patience:
            logger.info("Early stopping after %d non-improving epochs.", epochs_without_improvement)
            break

    if not best_model_dir.is_dir():
        raise RuntimeError("Training did not produce a best model checkpoint.")

    # Evaluate exactly once on the held-out test split after model selection.
    from src.model import load_finetuned_model

    best_tokenizer, best_model = load_finetuned_model(best_model_dir)
    best_model.to(device)
    test_loader = DataLoader(
        EssayScoringDataset(test_frame, best_tokenizer, config.max_length),
        batch_size=config.eval_batch_size,
        shuffle=False,
        **common_loader_args,
    )
    test_results = predict_loader(best_model, test_loader, device, use_amp=config.use_amp)
    test_metrics = evaluate_predictions(test_results)
    save_json(test_metrics, output_dir / "test_metrics.json")

    predictions = pd.DataFrame(
        {
            "row_id": test_results["row_ids"],
            "EssaySet": test_results["essay_sets"],
            "y_true": test_results["y_true"],
            "y_pred": test_results["y_pred"],
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    save_json(
        {
            "config": config.__dict__,
            "best_validation_qwk": None if best_qwk == -math.inf else best_qwk,
            "test_metrics": test_metrics,
            "note": "This is a new cleaned run; do not substitute historical write-up metrics.",
        },
        output_dir / "run_summary.json",
    )
    logger.info("Test metrics: %s", {k: v for k, v in test_metrics.items() if k != "classification_report"})
    logger.info("Training complete. Best model: %s", best_model_dir)


if __name__ == "__main__":
    main()
