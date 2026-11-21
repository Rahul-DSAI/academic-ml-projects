"""Command-line interface for PDF summarization."""

from __future__ import annotations

import argparse
from pathlib import Path

from .embeddings import BioBERTEmbedder
from .pdf import extract_pdf_text
from .summarizer import summarize_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract salient sentences from a medical PDF using BioBERT."
    )
    parser.add_argument("pdf", type=Path, help="Path to a text-based PDF")
    parser.add_argument("--sentences", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default="dmis-lab/biobert-v1.1")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sentences < 1:
        raise SystemExit("--sentences must be at least 1")

    text = extract_pdf_text(args.pdf)
    embedder = BioBERTEmbedder(model_name=args.model)
    summary = summarize_text(
        text,
        embedder,
        num_sentences=args.sentences,
        batch_size=args.batch_size,
    )

    print("\nEXTRACTIVE SUMMARY\n")
    print(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary + "\n", encoding="utf-8")
        print(f"\nSaved summary to: {args.output}")


if __name__ == "__main__":
    main()
