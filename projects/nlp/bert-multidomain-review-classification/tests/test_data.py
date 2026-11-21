from pathlib import Path

import pytest

from review_classifier.data import load_domain_reviews, make_splits


def test_loader_assigns_domain_ids(tmp_path):
    categories = ["books", "music"]
    for category in categories:
        folder = tmp_path / category
        folder.mkdir()
        (folder / "processed.review.balanced").write_text(
            "first review\nsecond review\nthird review\nfourth review\n", encoding="utf-8"
        )
    texts, labels, loaded_categories = load_domain_reviews(tmp_path, categories)
    assert len(texts) == 8
    assert labels == [0] * 4 + [1] * 4
    assert loaded_categories == categories


def test_stratified_splits_are_disjoint_and_reproducible():
    texts = [f"review-{i}" for i in range(100)]
    labels = [0] * 50 + [1] * 50
    first = make_splits(texts, labels, ["a", "b"], seed=12)
    second = make_splits(texts, labels, ["a", "b"], seed=12)
    assert first == second
    assert set(first.train_texts).isdisjoint(first.val_texts)
    assert set(first.train_texts).isdisjoint(first.test_texts)
    assert set(first.val_texts).isdisjoint(first.test_texts)
    assert set(first.train_labels) == {0, 1}
    assert set(first.val_labels) == {0, 1}
    assert set(first.test_labels) == {0, 1}


def test_loader_fails_for_missing_category_file(tmp_path):
    (tmp_path / "books").mkdir()
    with pytest.raises(FileNotFoundError):
        load_domain_reviews(tmp_path, ["books"])


def test_split_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        make_splits(["one"], [0, 1], ["a", "b"])
