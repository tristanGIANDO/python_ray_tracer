import json
from pathlib import Path
from dataclasses import dataclass

SCENE_CONFIG_FILE = Path("configs/scene_config.json")
RENDER_CONFIG_FILE = Path("configs/render_config.json")


@dataclass
class SceneConfig:
    data: dict


@dataclass
class RenderConfig:
    data: dict


def load_configs() -> tuple[dict, dict]:
    if not SCENE_CONFIG_FILE.exists():
        raise FileNotFoundError("Scene config file not found")

    if not RENDER_CONFIG_FILE.exists():
        raise FileNotFoundError("Render config file not found")

    scene_config = SceneConfig(json.load(open(SCENE_CONFIG_FILE)))
    render_config = RenderConfig(json.load(open(RENDER_CONFIG_FILE)))

    return scene_config, render_config
