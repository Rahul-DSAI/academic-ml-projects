"""Text normalization and sentence segmentation."""

from __future__ import annotations

import re
from typing import List


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace without deleting medical terms or stopwords."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK, with a regex fallback.

    NLTK's Punkt data is used when available. The fallback avoids requiring an
    extra NLTK download, although it is less robust for abbreviations.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            sentences = _fallback_split(text)
    except ImportError:
        sentences = _fallback_split(text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _fallback_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text)
    return [part.strip() for part in parts if part.strip()]
