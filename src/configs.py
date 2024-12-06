import json
from pathlib import Path
from dataclasses import dataclass

CONFIGS_ROOT = Path("configs")


@dataclass
class SceneConfig:
    data: dict


@dataclass
class RenderConfig:
    data: dict


def load_configs() -> tuple[dict, dict]:
    if not CONFIGS_ROOT.exists():
        raise FileNotFoundError("Config directory not found")

    scene_config_file = CONFIGS_ROOT / "scene_config.json"
    if not scene_config_file.exists():
        raise FileNotFoundError("Scene config file not found")

    render_config_file = CONFIGS_ROOT / "render_config.json"
    if not render_config_file.exists():
        raise FileNotFoundError("Render config file not found")

    scene_config = SceneConfig(json.load(open("configs/scene_config.json")))
    render_config = RenderConfig(json.load(open("configs/render_config.json")))

    return scene_config, render_config
