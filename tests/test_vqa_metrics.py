from gwel.data.vqa_metrics import exact_match, normalize_answer, vqa_accuracy


def test_normalize_lowercases_and_strips_articles() -> None:
    assert normalize_answer("The Red Car") == "red car"


def test_normalize_maps_number_words_and_contractions() -> None:
    assert normalize_answer("two") == "2"
    assert normalize_answer("dont know") == "don't know"


def test_normalize_strips_punctuation_but_keeps_decimals() -> None:
    assert normalize_answer("yes.") == "yes"
    assert normalize_answer("3.5") == "3.5"
    assert normalize_answer("stop!") == "stop"


def test_exact_match_uses_normalization() -> None:
    assert exact_match("The STOP sign.", ["stop sign"])
    assert not exact_match("go", ["stop"])


def test_vqa_accuracy_standard_thirds() -> None:
    answers = ["cat"] * 2 + ["dog"] * 8
    assert vqa_accuracy("cat", answers) == 2 / 3
    assert vqa_accuracy("dog", answers) == 1.0
    assert vqa_accuracy("bird", answers) == 0.0


def test_vqa_accuracy_degrades_to_exact_match_for_few_answers() -> None:
    assert vqa_accuracy("stop", ["stop"]) == 1.0
    assert vqa_accuracy("go", ["stop"]) == 0.0
    assert vqa_accuracy("anything", []) == 0.0
