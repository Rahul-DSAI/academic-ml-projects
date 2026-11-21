from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sklearn.model_selection import train_test_split

DEFAULT_CATEGORIES = [
    "apparel", "automotive", "baby", "beauty", "books", "camera_&_photo",
    "cell_phones_&_service", "computer_&_video_games", "dvd", "electronics",
    "gourmet_food", "grocery", "health_&_personal_care", "jewelry_&_watches",
    "kitchen_&_housewares", "magazines", "music", "musical_instruments",
    "office_products", "outdoor_living", "software", "sports_&_outdoors",
    "tools_&_hardware", "toys_&_games", "video",
]

@dataclass(frozen=True)
class ReviewSplits:
    train_texts: list[str]
    val_texts: list[str]
    test_texts: list[str]
    train_labels: list[int]
    val_labels: list[int]
    test_labels: list[int]
    categories: list[str]


def load_domain_reviews(
    data_root: str | Path,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    filename: str = "processed.review.balanced",
    min_samples_per_category: int = 3,
) -> tuple[list[str], list[int], list[str]]:
    """Load text files from <data_root>/<category>/<filename>.

    Each review is labelled with its product-domain folder. It does not infer
    positive/negative sentiment labels.
    """
    root = Path(data_root).expanduser()
    texts: list[str] = []
    labels: list[int] = []
    used_categories: list[str] = []

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    for class_id, category in enumerate(categories):
        file_path = root / category / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Expected review file not found: {file_path}")
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            category_reviews = [line.strip() for line in handle if line.strip()]
        if len(category_reviews) < min_samples_per_category:
            raise ValueError(
                f"Category '{category}' has {len(category_reviews)} non-empty reviews; "
                f"at least {min_samples_per_category} are required for stratified splitting."
            )
        texts.extend(category_reviews)
        labels.extend([class_id] * len(category_reviews))
        used_categories.append(category)

    if not texts:
        raise ValueError("No review text was loaded.")
    return texts, labels, used_categories


def make_splits(
    texts: Sequence[str],
    labels: Sequence[int],
    categories: Sequence[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> ReviewSplits:
    """Create reproducible stratified train/validation/test splits."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be > 0 and sum to < 1")

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        list(texts), list(labels), test_size=test_size, random_state=seed, stratify=list(labels)
    )
    relative_val = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=relative_val,
        random_state=seed, stratify=y_train_val,
    )
    return ReviewSplits(
        train_texts=x_train, val_texts=x_val, test_texts=x_test,
        train_labels=y_train, val_labels=y_val, test_labels=y_test,
        categories=list(categories),
    )
