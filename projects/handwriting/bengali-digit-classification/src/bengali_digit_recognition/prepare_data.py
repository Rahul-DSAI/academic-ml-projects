"""Prepare the source class folders into reproducible train/val/test splits."""

from __future__ import annotations

import argparse

from .data import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Directory containing class folders 0..9")
    parser.add_argument("--output", default="data/processed", help="Output split directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print("Prepared dataset splits:")
    for split_name, count in counts.items():
        print(f"  {split_name}: {count} images")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
