from src.data.splits import ratio_split_indices


def test_ratio_split_indices_are_disjoint_and_reproducible():
    first = ratio_split_indices(70, seed=42, train_ratio=0.7, eval_ratio=0.3)
    second = ratio_split_indices(70, seed=42, train_ratio=0.7, eval_ratio=0.3)

    assert first == second
    assert len(first["train"]) == 49
    assert len(first["eval"]) == 21
    assert len(first["test"]) == 0
    assert set(first["train"]).isdisjoint(first["eval"])


if __name__ == "__main__":
    test_ratio_split_indices_are_disjoint_and_reproducible()

