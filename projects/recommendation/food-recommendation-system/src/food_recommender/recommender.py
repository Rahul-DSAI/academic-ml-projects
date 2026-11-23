from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATALOG_REQUIRED = {
    "foodCategoryName", "foodItemName", "foodInfo", "restaurantName", "tags",
    "Dietary", "FoodPreparationStyles", "FoodPresentation", "FoodConcepts",
    "Flavortype", "Mealtime", "UserRating",
}
PREFERENCE_REQUIRED = CATALOG_REQUIRED - {"UserRating"}
TEXT_FIELDS = [
    "foodCategoryName", "foodItemName", "foodInfo", "tags", "Dietary",
    "FoodPreparationStyles", "FoodPresentation", "FoodConcepts", "Flavortype", "Mealtime",
]
DISPLAY_COLUMNS = [
    "foodCategoryName", "foodItemName", "restaurantName", "UserRating", "Similarity_Score"
]


def _check_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def load_catalog(path: str | Path) -> pd.DataFrame:
    """Load the menu catalogue and validate the expected schema."""
    frame = pd.read_csv(path)
    _check_columns(frame, CATALOG_REQUIRED, "Food catalogue")
    if frame.empty:
        raise ValueError("Food catalogue is empty.")
    frame = frame.copy()
    frame["UserRating"] = pd.to_numeric(frame["UserRating"], errors="coerce")
    return frame


def load_preferences(path: str | Path) -> pd.DataFrame:
    """Load the preference examples and validate the expected schema."""
    frame = pd.read_csv(path)
    _check_columns(frame, PREFERENCE_REQUIRED, "User preferences")
    if frame.empty:
        raise ValueError("User preference file is empty.")
    return frame.copy()


def _clean_value(value: object) -> str:
    if pd.isna(value):
        return ""
    # Comma-separated metadata is treated as a list of descriptive terms.
    text = str(value).replace(",", " ").replace("&", " and ")
    return re.sub(r"\s+", " ", text).strip().lower()


def _row_to_profile(row: pd.Series) -> str:
    parts = []
    for field in TEXT_FIELDS:
        value = _clean_value(row.get(field, ""))
        if value:
            parts.append(f"{field} {value}")
    return " ".join(parts)


class FoodRecommender:
    """TF-IDF/cosine content-based recommender using menu attributes."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=ngram_range,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self.catalog: pd.DataFrame | None = None
        self.catalog_matrix = None
        self.preferences: pd.DataFrame | None = None
        self.preference_matrix = None

    def fit(self, catalog: pd.DataFrame, preferences: pd.DataFrame) -> "FoodRecommender":
        _check_columns(catalog, CATALOG_REQUIRED, "Food catalogue")
        _check_columns(preferences, PREFERENCE_REQUIRED, "User preferences")
        if catalog.empty or preferences.empty:
            raise ValueError("Both catalogue and preference data must contain at least one row.")
        self.catalog = catalog.reset_index(drop=True).copy()
        self.preferences = preferences.reset_index(drop=True).copy()
        catalog_profiles = self.catalog.apply(_row_to_profile, axis=1).tolist()
        preference_profiles = self.preferences.apply(_row_to_profile, axis=1).tolist()
        # Fit on both catalog and preference text so unseen preference terms are retained.
        self.vectorizer.fit(catalog_profiles + preference_profiles)
        self.catalog_matrix = self.vectorizer.transform(catalog_profiles)
        self.preference_matrix = self.vectorizer.transform(preference_profiles)
        return self

    def recommend(
        self,
        preferred_foods: str | Sequence[str] | None = None,
        top_n: int = 10,
        exclude_preferred: bool = True,
    ) -> pd.DataFrame:
        """Rank catalogue items against one or more named preference examples.

        When ``preferred_foods`` is omitted, all rows in the supplied preference
        file are averaged into a single preference profile.
        """
        if self.catalog is None or self.catalog_matrix is None or self.preferences is None or self.preference_matrix is None:
            raise RuntimeError("Call fit(catalog, preferences) before recommend().")
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        if preferred_foods is None:
            pref_indices = list(range(len(self.preferences)))
            excluded_names = set(self.preferences["foodItemName"].astype(str))
        else:
            names = [preferred_foods] if isinstance(preferred_foods, str) else list(preferred_foods)
            if not names:
                raise ValueError("Provide at least one preferred food name.")
            pref_names = self.preferences["foodItemName"].astype(str)
            pref_indices = []
            missing = []
            for name in names:
                matches = self.preferences.index[pref_names.str.casefold() == str(name).casefold()].tolist()
                if not matches:
                    missing.append(str(name))
                else:
                    pref_indices.extend(matches)
            if missing:
                available = ", ".join(self.preferences["foodItemName"].astype(str).tolist())
                raise ValueError(f"Preferred food not found in user_preferences.csv: {', '.join(missing)}. Available examples: {available}")
            excluded_names = set(map(str, names))

        # Combine preference vectors into a single centroid/profile.
        query_matrix = np.asarray(self.preference_matrix[pref_indices].mean(axis=0)).reshape(1, -1)
        scores = cosine_similarity(self.catalog_matrix, query_matrix).ravel()
        result = self.catalog.copy()
        result["Similarity_Score"] = scores
        if exclude_preferred:
            result = result[~result["foodItemName"].astype(str).str.casefold().isin({n.casefold() for n in excluded_names})]
        result = result.sort_values(
            ["Similarity_Score", "UserRating", "foodItemName"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(top_n)
        columns = [c for c in DISPLAY_COLUMNS if c in result.columns]
        return result[columns].reset_index(drop=True)
