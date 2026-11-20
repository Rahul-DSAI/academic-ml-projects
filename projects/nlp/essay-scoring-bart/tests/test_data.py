import pandas as pd
import pytest

from src.data import load_data, split_data


def make_frame(n_per_class: int = 10) -> pd.DataFrame:
    rows = []
    row_id = 1
    for label in range(4):
        for _ in range(n_per_class):
            rows.append(
                {
                    "Id": row_id,
                    "EssaySet": (row_id % 3) + 1,
                    "Score1": label,
                    "Score2": label,
                    "EssayText": f"Example response {row_id}",
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def test_load_data_reads_expected_tsv(tmp_path):
    path = tmp_path / "train.tsv"
    make_frame().to_csv(path, sep="\t", index=False)
    frame = load_data(path)
    assert len(frame) == 40
    assert set(frame["Score1"].unique()) == {0, 1, 2, 3}


def test_load_data_rejects_unknown_label(tmp_path):
    path = tmp_path / "train.tsv"
    frame = make_frame()
    frame.loc[0, "Score1"] = 4
    frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="Unexpected Score1 labels"):
        load_data(path)


def test_split_data_is_reproducible_and_disjoint():
    frame = make_frame(25)
    train, validation, test = split_data(frame, 0.1, 0.1, 42)
    assert set(train["Id"]).isdisjoint(set(validation["Id"]))
    assert set(train["Id"]).isdisjoint(set(test["Id"]))
    assert set(validation["Id"]).isdisjoint(set(test["Id"]))
    assert len(train) + len(validation) + len(test) == len(frame)
    train2, validation2, test2 = split_data(frame, 0.1, 0.1, 42)
    assert train["Id"].tolist() == train2["Id"].tolist()
    assert validation["Id"].tolist() == validation2["Id"].tolist()
    assert test["Id"].tolist() == test2["Id"].tolist()
