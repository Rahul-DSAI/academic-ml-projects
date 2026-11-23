"""Content-based food recommendation system."""
from .recommender import FoodRecommender, load_catalog, load_preferences

__all__ = ["FoodRecommender", "load_catalog", "load_preferences"]
