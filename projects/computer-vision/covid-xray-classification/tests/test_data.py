from pathlib import Path

import pytest

from covid_xray.data import scan_image_dataset, stratified_split, write_manifest, read_manifest


def make_dataset(root: Path):
    for label in ("COVID", "Normal", "Viral Pneumonia"):
        directory = root / label
        directory.mkdir(parents=True)
        for index in range(20):
            (directory / f"image-{index:02d}.png").write_bytes(f"{label}-{index}".encode())


def test_scan_and_counts(tmp_path):
    make_dataset(tmp_path)
    records, audit = scan_image_dataset(tmp_path)
    assert len(records) == 60
    assert audit["unique_image_count"] == 60
    assert set(audit["class_names"]) == {"COVID", "Normal", "Viral Pneumonia"}


def test_exact_duplicates_are_removed(tmp_path):
    make_dataset(tmp_path)
    original = tmp_path / "COVID" / "image-00.png"
    duplicate = tmp_path / "COVID" / "duplicate.png"
    duplicate.write_bytes(original.read_bytes())
    records, audit = scan_image_dataset(tmp_path)
    assert len(records) == 60
    assert audit["exact_duplicate_count"] == 1


def test_identical_content_across_classes_is_rejected(tmp_path):
    make_dataset(tmp_path)
    payload = b"same image content"
    (tmp_path / "COVID" / "cross-class.png").write_bytes(payload)
    (tmp_path / "Normal" / "cross-class.png").write_bytes(payload)
    with pytest.raises(ValueError, match="different class labels"):
        scan_image_dataset(tmp_path)


def test_stratified_splits_are_reproducible_and_disjoint(tmp_path):
    make_dataset(tmp_path)
    records, _ = scan_image_dataset(tmp_path)
    rows_a = stratified_split(records, seed=17)
    rows_b = stratified_split(records, seed=17)
    assert [(r["relative_path"], r["split"]) for r in rows_a] == [
        (r["relative_path"], r["split"]) for r in rows_b
    ]
    paths_by_split = {
        split: {r["relative_path"] for r in rows_a if r["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert not (paths_by_split["train"] & paths_by_split["validation"])
    assert not (paths_by_split["train"] & paths_by_split["test"])
    assert not (paths_by_split["validation"] & paths_by_split["test"])
    for class_name in ("COVID", "Normal", "Viral Pneumonia"):
        assert {r["split"] for r in rows_a if r["class_name"] == class_name} == {"train", "validation", "test"}


def test_manifest_round_trip(tmp_path):
    make_dataset(tmp_path / "dataset")
    records, _ = scan_image_dataset(tmp_path / "dataset")
    rows = stratified_split(records)
    manifest = write_manifest(rows, tmp_path / "manifest.csv")
    loaded = read_manifest(manifest, split="test")
    assert loaded
    assert all(row["split"] == "test" for row in loaded)
    assert all(isinstance(row["label"], int) for row in loaded)
