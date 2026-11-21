"""Sentence-centrality extractive summarization."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .text import split_sentences


class Embedder(Protocol):
    def encode(self, sentences: Sequence[str], batch_size: int = 16) -> np.ndarray:
        ...


def rank_sentences(embeddings: np.ndarray) -> np.ndarray:
    """Score sentences by mean cosine similarity to all other sentences."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    count = embeddings.shape[0]
    if count == 0:
        return np.empty((0,), dtype=np.float32)
    if count == 1:
        return np.ones((1,), dtype=np.float32)

    similarity = cosine_similarity(embeddings)
    np.fill_diagonal(similarity, 0.0)
    return similarity.sum(axis=1) / (count - 1)


def select_summary_sentences(
    sentences: Sequence[str],
    embeddings: np.ndarray,
    num_sentences: int = 3,
) -> list[str]:
    """Select central sentences and return them in original document order."""
    if num_sentences < 1:
        raise ValueError("num_sentences must be at least 1")
    if len(sentences) != len(embeddings):
        raise ValueError("sentences and embeddings must have the same length")
    if not sentences:
        return []

    scores = rank_sentences(embeddings)
    selected_indices = np.argsort(scores)[-min(num_sentences, len(sentences)):]
    selected_indices = sorted(int(index) for index in selected_indices)
    return [sentences[index] for index in selected_indices]


def summarize_text(
    text: str,
    embedder: Embedder,
    num_sentences: int = 3,
    batch_size: int = 16,
) -> str:
    """Create an extractive summary using an injected sentence embedder."""
    if num_sentences < 1:
        raise ValueError("num_sentences must be at least 1")

    sentences = split_sentences(text)
    if not sentences:
        return ""

    embeddings = embedder.encode(sentences, batch_size=batch_size)
    chosen = select_summary_sentences(sentences, embeddings, num_sentences)
    return " ".join(chosen)
