"""Dataset loading, stratified splitting and PyTorch dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

REQUIRED_COLUMNS = {"EssayText", "Score1", "EssaySet"}
VALID_LABELS = {0, 1, 2, 3}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the ASAP-SAS training TSV and validate required fields.

    The original project trains against Score1. Score2 is retained in the
    source dataset for inter-rater/reliability work but is not used as target.
    """
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Obtain train.tsv under the dataset's "
            "applicable terms and place it at the configured data_path."
        )

    frame = pd.read_csv(data_path, sep="\t")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["EssayText"] = frame["EssayText"].fillna("").astype(str)
    frame["Score1"] = pd.to_numeric(frame["Score1"], errors="coerce")
    frame["EssaySet"] = pd.to_numeric(frame["EssaySet"], errors="coerce")
    frame = frame.dropna(subset=["Score1", "EssaySet"])

    if frame.empty:
        raise ValueError("No usable rows remain after removing missing labels or essay sets.")

    if not (frame["Score1"] % 1 == 0).all():
        raise ValueError("Score1 contains non-integer values; expected classes 0, 1, 2 and 3.")
    frame["Score1"] = frame["Score1"].astype(int)
    frame["EssaySet"] = frame["EssaySet"].astype(int)

    unexpected = sorted(set(frame["Score1"].unique()) - VALID_LABELS)
    if unexpected:
        raise ValueError(f"Unexpected Score1 labels: {unexpected}; expected only 0, 1, 2, 3.")

    if (frame["EssayText"].str.strip() == "").any():
        frame = frame.loc[frame["EssayText"].str.strip() != ""].copy()

    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("No non-empty essay responses were found.")
    return frame


def split_data(
    frame: pd.DataFrame,
    test_size: float = 0.1,
    validation_size: float = 0.1,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits stratified by Score1.

    validation_size and test_size are proportions of the complete dataset.
    This is a row-level random split, matching the general approach in the
    original coursework code; it is not a held-out-prompt evaluation.
    """
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("Both split sizes must be positive and their sum must be below 1.")

    counts = frame["Score1"].value_counts()
    if counts.min() < 3:
        raise ValueError("At least three examples per class are required for stratified train/val/test splits.")

    train_validation, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_seed,
        stratify=frame["Score1"],
    )
    validation_fraction_of_remainder = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction_of_remainder,
        random_state=random_seed,
        stratify=train_validation["Score1"],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


class EssayScoringDataset(Dataset):
    """Tokenise student responses for BART sequence classification."""

    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int = 512) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        row = self.frame.iloc[index]
        text = str(row["EssayText"])
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["Score1"]), dtype=torch.long),
            "row_id": int(row["Id"]) if "Id" in row and pd.notna(row["Id"]) else index,
            "essay_set": int(row["EssaySet"]),
        }


class DynamicPaddingCollator:
    """Pad each batch to its longest sequence rather than always padding to 512."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        labels = torch.stack([feature["labels"] for feature in features])
        row_ids = [feature["row_id"] for feature in features]
        essay_sets = [feature["essay_set"] for feature in features]
        token_features = [
            {"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]}
            for feature in features
        ]
        batch = self.tokenizer.pad(token_features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        batch["row_ids"] = row_ids
        batch["essay_sets"] = essay_sets
        return batch
