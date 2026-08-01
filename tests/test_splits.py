import pytest

from gwel.router.splits import make_split


def _ids(n: int, prefix: str = "ex") -> list[str]:
    return [f"{prefix}-{i:03d}" for i in range(n)]


def test_folds_partition_every_example() -> None:
    ids = _ids(100)
    split = make_split(ids)
    assert sorted([*split.train, *split.val, *split.test]) == sorted(ids)
    assert split.sizes["val"] == 20
    assert split.sizes["test"] == 20


def test_assignment_is_stable_across_calls() -> None:
    ids = _ids(50)
    assert make_split(ids) == make_split(list(reversed(ids)))


def test_adding_examples_does_not_reshuffle_existing_ones() -> None:
    # Hash-based assignment means a grown pilot stays comparable to the old one.
    small = make_split(_ids(200), val_fraction=0.0, test_fraction=0.5)
    large = make_split(_ids(400), val_fraction=0.0, test_fraction=0.5)
    moved = sum(
        1
        for example in _ids(200)
        if (example in small.test) != (example in large.test)
    )
    assert moved < 40  # a reshuffle would move about half of them


def test_stratification_keeps_every_dataset_in_every_fold() -> None:
    ids = _ids(40, "a") + _ids(40, "b")
    datasets = ["docvqa"] * 40 + ["textvqa"] * 40
    split = make_split(ids, datasets)
    for fold in (split.train, split.val, split.test):
        prefixes = {e.split("-")[0] for e in fold}
        assert prefixes == {"a", "b"}


def test_different_seeds_give_different_assignments() -> None:
    ids = _ids(100)
    assert make_split(ids, seed=1) != make_split(ids, seed=2)


def test_fold_lookup_and_unknown_example() -> None:
    split = make_split(_ids(20))
    assert split.fold_of(split.train[0]) == "train"
    with pytest.raises(KeyError):
        split.fold_of("not-an-example")


def test_rejects_impossible_fractions() -> None:
    with pytest.raises(ValueError):
        make_split(_ids(10), val_fraction=0.6, test_fraction=0.6)
    with pytest.raises(ValueError):
        make_split(_ids(10), val_fraction=-0.1)


def test_rejects_mismatched_dataset_labels() -> None:
    with pytest.raises(ValueError):
        make_split(_ids(10), ["docvqa"] * 3)
