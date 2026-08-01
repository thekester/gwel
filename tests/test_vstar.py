from gwel.data.loaders import _vstar_gold_answers


def test_short_answer_extracted_from_templated_options() -> None:
    options = [
        "The color of the flag is white.",
        "The color of the flag is red.",
    ]
    assert _vstar_gold_answers(options) == ("white", "The color of the flag is white.")


def test_multiword_answer_and_many_options() -> None:
    options = [
        "The shop is a yoga studio.",
        "The shop is a cafe.",
        "The shop is a seven-eleven.",
        "The shop is a milk tea shop.",
    ]
    assert _vstar_gold_answers(options)[0] == "yoga studio"


def test_single_option_returned_as_is() -> None:
    assert _vstar_gold_answers(["Just one answer."]) == ("Just one answer.",)


def test_options_without_common_prefix_fall_back_to_full_sentence() -> None:
    answers = _vstar_gold_answers(["Yes, it is.", "No, it is not."])
    assert "Yes, it is." in answers
