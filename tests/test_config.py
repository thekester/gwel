import pytest

from gwel.config import GwelConfig, load_config


def test_default_config_file_loads(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    assert config.model.model_id == "HuggingFaceTB/SmolVLM-500M-Instruct"
    assert config.runner.lowres_sizes == (384, 768)
    assert config.runner.crop.rows == 2
    assert config.cost.error_weight == 1.0


def test_empty_yaml_gives_defaults(tmp_path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == GwelConfig()


def test_partial_override_keeps_other_defaults(tmp_path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("runner:\n  repeats: 5\n", encoding="utf-8")
    config = load_config(path)
    assert config.runner.repeats == 5
    assert config.runner.lowres_sizes == GwelConfig().runner.lowres_sizes


def test_unknown_key_rejected(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("runner:\n  typo_key: 1\n", encoding="utf-8")
    with pytest.raises(KeyError, match="typo_key"):
        load_config(path)
