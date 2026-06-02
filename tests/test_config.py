from pathlib import Path

from src.config.config import SIMAConfig, get_baseline_config, load_config, save_config
from src.main import create_components


def test_config_round_trip_json(tmp_path: Path):
    """Configuration should survive save/load without losing runtime fields."""
    config = get_baseline_config()
    config.training.success_threshold = 7.5
    config.minecraft.use_minerl = False

    config_path = tmp_path / "config.json"
    save_config(config, config_path)
    loaded = load_config(config_path)

    assert loaded.training.success_threshold == 7.5
    assert loaded.minecraft.use_minerl is False
    assert loaded.experiment_name == config.experiment_name


def test_create_components_uses_success_threshold():
    """Agent creation should wire the training success threshold through."""
    config = SIMAConfig()
    config.training.success_threshold = 9.0

    agent, _, _, _, _ = create_components(config)

    assert agent.success_threshold == 9.0
