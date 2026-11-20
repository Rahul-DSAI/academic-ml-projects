"""Evaluate a saved BART checkpoint on the deterministic test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data import DynamicPaddingCollator, EssayScoringDataset, load_data, split_data
from src.engine import evaluate_predictions, predict_loader
from src.model import load_finetuned_model
from src.utils import get_device, make_logger, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--model-dir", default="artifacts/best_model")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    data_path = args.data or config.data_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger("essay_scoring.evaluate")
    set_seed(config.random_seed)

    frame = load_data(data_path)
    _, _, test_frame = split_data(
        frame,
        test_size=config.test_size,
        validation_size=config.validation_size,
        random_seed=config.random_seed,
    )

    tokenizer, model = load_finetuned_model(args.model_dir)
    device = get_device()
    model.to(device)
    loader = DataLoader(
        EssayScoringDataset(test_frame, tokenizer, config.max_length),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=DynamicPaddingCollator(tokenizer),
    )
    results = predict_loader(model, loader, device, use_amp=config.use_amp)
    metrics = evaluate_predictions(results)
    save_json(metrics, output_dir / "metrics.json")
    pd.DataFrame(
        {
            "row_id": results["row_ids"],
            "EssaySet": results["essay_sets"],
            "y_true": results["y_true"],
            "y_pred": results["y_pred"],
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    logger.info("Metrics: %s", {k: v for k, v in metrics.items() if k != "classification_report"})


if __name__ == "__main__":
    main()
