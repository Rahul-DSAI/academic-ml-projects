from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from food_recommender.recommender import FoodRecommender, load_catalog, load_preferences

ROOT = Path(__file__).resolve().parents[1]


def sample_data():
    catalog = load_catalog(ROOT / "data" / "food_data.csv")
    preferences = load_preferences(ROOT / "data" / "user_preferences.csv")
    return catalog, preferences


def test_original_catalogue_and_preferences_load():
    catalog, preferences = sample_data()
    assert len(catalog) == 38
    assert len(preferences) == 13
    assert catalog["foodItemName"].is_unique


def test_recommender_returns_ranked_rows_and_scores():
    catalog, preferences = sample_data()
    result = FoodRecommender().fit(catalog, preferences).recommend("Lunch Sashimi", top_n=5)
    assert len(result) == 5
    assert "Similarity_Score" in result.columns
    assert result["Similarity_Score"].between(0, 1).all()
    assert "Lunch Sashimi" not in result["foodItemName"].tolist()
    assert result["Similarity_Score"].is_monotonic_decreasing


def test_all_preferences_can_be_used_as_profile():
    catalog, preferences = sample_data()
    result = FoodRecommender().fit(catalog, preferences).recommend(top_n=10)
    assert len(result) <= 10
    assert "Similarity_Score" in result.columns


def test_unknown_preference_has_actionable_error():
    catalog, preferences = sample_data()
    recommender = FoodRecommender().fit(catalog, preferences)
    with pytest.raises(ValueError, match="Preferred food not found"):
        recommender.recommend("Not A Real Dish")


def test_top_n_validation():
    catalog, preferences = sample_data()
    recommender = FoodRecommender().fit(catalog, preferences)
    with pytest.raises(ValueError, match="top_n"):
        recommender.recommend("Lunch Sashimi", top_n=0)
