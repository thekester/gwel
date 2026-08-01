import pytest

from gwel.profiling.coldstart import TOOL_SNIPPETS, ColdStartReport, measure_cold_start


def test_measures_a_trivial_tool_in_a_fresh_process() -> None:
    report = measure_cold_start(
        "toy", snippet="def init():\n    import json\n    return json.dumps({})\n"
    )
    assert report.error is None
    assert report.cold_ms is not None and report.cold_ms > 0
    assert report.warm_ms is not None
    assert report.warm_ms <= report.cold_ms  # the second call reuses the import


def test_a_failing_snippet_reports_the_error_rather_than_raising() -> None:
    report = measure_cold_start(
        "broken", snippet="def init():\n    raise RuntimeError('nope')\n"
    )
    assert report.error is not None
    assert report.cold_ms is None


def test_a_hanging_snippet_times_out() -> None:
    report = measure_cold_start(
        "slow",
        snippet="def init():\n    import time\n    time.sleep(30)\n",
        timeout_s=2.0,
    )
    assert report.error is not None and "timeout" in report.error


def test_unknown_tool_without_a_snippet_is_rejected() -> None:
    with pytest.raises(KeyError):
        measure_cold_start("not-a-tool")


def test_known_tools_are_registered() -> None:
    assert {"pytesseract", "easyocr", "smolvlm"} <= set(TOOL_SNIPPETS)
    assert "{model_id}" in TOOL_SNIPPETS["smolvlm"]  # takes a format argument


def test_report_serialises_every_field() -> None:
    payload = ColdStartReport("x", 1.0, 2.0, 3.0).to_dict()
    assert set(payload) == {"tool", "cold_ms", "warm_ms", "ram_delta_mb", "error"}
