import numpy as np
import pytest

from medical_summarizer.summarizer import (
    rank_sentences,
    select_summary_sentences,
    summarize_text,
)


def test_rank_sentences_shape_and_values():
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    scores = rank_sentences(embeddings)
    assert scores.shape == (3,)
    assert scores[0] > scores[2]


def test_select_summary_sentences_preserves_original_order():
    sentences = ["First sentence.", "Middle sentence.", "Last sentence."]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    result = select_summary_sentences(sentences, embeddings, num_sentences=2)
    assert result == ["First sentence.", "Middle sentence."]


def test_single_sentence():
    result = select_summary_sentences(["Only sentence."], np.array([[1.0, 0.0]]))
    assert result == ["Only sentence."]


def test_invalid_num_sentences():
    with pytest.raises(ValueError):
        select_summary_sentences(["A."], np.array([[1.0, 0.0]]), num_sentences=0)


class DummyEmbedder:
    def encode(self, sentences, batch_size=16):
        vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
        return vectors[:len(sentences)]


def test_summarize_text_with_dummy_embedder():
    text = "First sentence. Second sentence. Third sentence."
    summary = summarize_text(text, DummyEmbedder(), num_sentences=2)
    assert summary
    assert summary.count(".") == 2
