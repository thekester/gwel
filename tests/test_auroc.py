import math

import pytest

from gwel.router.evaluate import auroc


def test_perfect_ranking_gives_one() -> None:
    assert auroc([0.1, 0.2, 0.9, 0.95], [False, False, True, True]) == 1.0


def test_inverted_ranking_gives_zero() -> None:
    assert auroc([0.9, 0.95, 0.1, 0.2], [False, False, True, True]) == 0.0


def test_constant_score_gives_half() -> None:
    assert auroc([0.5] * 4, [True, False, True, False]) == pytest.approx(0.5)


def test_ties_between_classes_count_as_half() -> None:
    # One positive tied with one negative, one positive ranked above.
    assert auroc([1.0, 1.0, 2.0], [False, True, True]) == pytest.approx(0.75)


def test_single_class_is_undefined() -> None:
    assert math.isnan(auroc([0.1, 0.9], [True, True]))
    assert math.isnan(auroc([0.1, 0.9], [False, False]))


def test_validates_input_shapes() -> None:
    with pytest.raises(ValueError):
        auroc([0.1], [True, False])
    with pytest.raises(ValueError):
        auroc([], [])
