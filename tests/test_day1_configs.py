"""Day 1 tests — just confirm the configs are valid and consistent.
Real detector/pipeline tests start once src/ has real modules (Day 5+)."""
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
MISSIONS_DIR = ROOT / "configs" / "missions"


def test_main_config_loads():
    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)
    assert config["project"]["name"] == "GaganMitra"
    assert "windowing" in config
    assert "severity" in config


def test_all_mission_configs_have_required_fields():
    mission_files = list(MISSIONS_DIR.glob("*.yaml"))
    assert len(mission_files) >= 2, "expected at least the ESA-ADB and synthetic configs"
    for path in mission_files:
        with open(path) as f:
            mission = yaml.safe_load(f)
        assert "mission_id" in mission
        assert "source" in mission
        assert "channels" in mission
        assert len(mission["channels"]) >= 1


def test_default_mission_config_exists():
    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)
    default = config["default_mission"]
    assert (MISSIONS_DIR / f"{default}.yaml").exists(), (
        f"config.yaml points to default_mission '{default}' but no matching "
        f"file exists in configs/missions/"
    )
