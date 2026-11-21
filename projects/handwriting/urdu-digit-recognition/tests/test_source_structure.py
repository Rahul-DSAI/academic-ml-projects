from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_both_tracks_have_training_scripts():
    assert (ROOT / "src/urdu_recognition/characters_tf/train.py").is_file()
    assert (ROOT / "src/urdu_recognition/digits_torch/train.py").is_file()


def test_original_sources_are_preserved():
    assert (ROOT / "experiments/original/urdhumnist.py").is_file()
    assert (ROOT / "experiments/original/urdu_pytorch.py").is_file()


def test_readme_explains_distinct_tracks():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "TensorFlow" in readme
    assert "PyTorch" in readme
    assert "40-class" in readme
    assert "10-class" in readme
