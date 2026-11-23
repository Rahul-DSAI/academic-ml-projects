from __future__ import annotations

import argparse
from pathlib import Path

from .recommender import FoodRecommender, load_catalog, load_preferences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recommend similar food items using TF-IDF and cosine similarity.")
    parser.add_argument("--catalog", default="data/food_data.csv", help="Path to food catalogue CSV")
    parser.add_argument("--preferences", default="data/user_preferences.csv", help="Path to user preference CSV")
    parser.add_argument("--preferred-food", action="append", help="Preference item name; may be supplied more than once")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to return")
    parser.add_argument("--include-preferred", action="store_true", help="Allow selected preference items to appear in results")
    parser.add_argument("--output", help="Optional output CSV path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog = load_catalog(Path(args.catalog))
    preferences = load_preferences(Path(args.preferences))
    recommender = FoodRecommender().fit(catalog, preferences)
    recommendations = recommender.recommend(
        preferred_foods=args.preferred_food,
        top_n=args.top_n,
        exclude_preferred=not args.include_preferred,
    )
    if recommendations.empty:
        print("No recommendations found. Check the preference items and catalogue.")
    else:
        print(recommendations.to_string(index=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(output_path, index=False)
        print(f"\nSaved recommendations to {output_path}")


if __name__ == "__main__":
    main()
