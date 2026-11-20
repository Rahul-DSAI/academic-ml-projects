"""Training and inference loops."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics import calculate_metrics


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _make_grad_scaler(enabled: bool):
    """Support both newer and older supported PyTorch GradScaler APIs."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    scaler = _make_grad_scaler(enabled=use_amp and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    number_of_batches = len(loader)

    progress = tqdm(loader, desc="Training", leave=False)
    for batch_index, batch in enumerate(progress):
        labels = batch["labels"].to(device)
        model_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": labels,
        }

        with _autocast_context(device, use_amp):
            outputs = model(**model_inputs)
            loss = outputs.loss

        # Average gradients inside each accumulation group. This also handles
        # the final group when the number of batches is not divisible by N.
        group_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
        group_size = min(gradient_accumulation_steps, number_of_batches - group_start)
        scaler.scale(loss / group_size).backward()

        predictions = outputs.logits.argmax(dim=-1)
        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_examples += batch_size

        is_accumulation_step = (batch_index + 1) % gradient_accumulation_steps == 0
        is_last_batch = batch_index + 1 == number_of_batches
        if is_accumulation_step or is_last_batch:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_examples == 0:
        raise RuntimeError("Training dataloader produced no batches.")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.inference_mode()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []
    all_ids: list[int] = []
    all_essay_sets: list[int] = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        labels = batch["labels"].to(device)
        model_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        with _autocast_context(device, use_amp):
            outputs = model(**model_inputs)
        all_logits.append(outputs.logits.float().cpu().numpy())
        all_labels.extend(labels.cpu().tolist())
        all_ids.extend(batch["row_ids"])
        all_essay_sets.extend(batch["essay_sets"])

    if not all_logits:
        raise RuntimeError("Evaluation dataloader produced no batches.")

    logits = np.concatenate(all_logits, axis=0)
    predictions = logits.argmax(axis=1)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return {
        "logits": logits,
        "probabilities": probabilities,
        "y_true": np.asarray(all_labels, dtype=int),
        "y_pred": predictions.astype(int),
        "row_ids": all_ids,
        "essay_sets": all_essay_sets,
    }


def evaluate_predictions(results: dict[str, Any]) -> dict:
    return calculate_metrics(results["y_true"], results["y_pred"])


def optimizer_steps_per_epoch(number_of_batches: int, accumulation_steps: int) -> int:
    return math.ceil(number_of_batches / accumulation_steps)
