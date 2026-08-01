from gwel.oracle.records import deduplicate_records


def test_last_measurement_of_a_key_wins(make_record) -> None:
    stale = make_record(config_id="lowres_256", visual_tokens=922)
    fresh = make_record(config_id="lowres_256", visual_tokens=64)
    assert deduplicate_records([stale, fresh]) == [fresh]


def test_distinct_keys_are_all_kept_in_order(make_record) -> None:
    records = [
        make_record(example_id="a", config_id="lowres_256"),
        make_record(example_id="a", config_id="full"),
        make_record(example_id="b", config_id="lowres_256"),
    ]
    assert deduplicate_records(records) == records


def test_duplicate_removal_preserves_first_appearance_order(make_record) -> None:
    first = make_record(example_id="a", config_id="lowres_256")
    second = make_record(example_id="b", config_id="lowres_256")
    repeat = make_record(example_id="a", config_id="lowres_256", visual_tokens=1)
    assert [r.key for r in deduplicate_records([first, second, repeat])] == [
        ("a", "lowres_256"),
        ("b", "lowres_256"),
    ]


def test_empty_input() -> None:
    assert deduplicate_records([]) == []
