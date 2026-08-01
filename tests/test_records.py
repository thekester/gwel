import pytest

from gwel.oracle.records import RunRecord, append_records, load_done_keys, read_records


def test_record_jsonl_round_trip(tmp_path, make_record) -> None:
    records = [make_record(config_id="lowres_256"), make_record(config_id="crop_r0c0")]
    path = tmp_path / "records.jsonl"
    append_records(path, records)
    assert read_records(path) == records


def test_append_is_incremental_and_resume_keys_match(tmp_path, make_record) -> None:
    path = tmp_path / "records.jsonl"
    append_records(path, [make_record(example_id="a", config_id="lowres_256")])
    append_records(path, [make_record(example_id="a", config_id="full", action=None)])
    assert load_done_keys(path) == {("a", "lowres_256"), ("a", "full")}


def test_done_keys_empty_for_missing_file(tmp_path) -> None:
    assert load_done_keys(tmp_path / "absent.jsonl") == set()


def test_read_records_rejects_malformed_lines(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"example_id": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_records(path)


def test_record_preserves_null_measurements(tmp_path, make_record) -> None:
    record = make_record(
        energy_total_mj=None,
        ttft_ms=None,
        vram_peak_mb=None,
        energy_mj={"total": None},
    )
    path = tmp_path / "records.jsonl"
    append_records(path, [record])
    loaded = read_records(path)[0]
    assert loaded.total_energy_mj is None
    assert loaded.ttft_ms is None
    assert loaded.vram_peak_mb is None
