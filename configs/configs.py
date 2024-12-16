import json
from dataclasses import dataclass
from pathlib import Path

SCENE_CONFIG_FILE = Path("configs/scene_config.json")
RENDER_CONFIG_FILE = Path("configs/render_config.json")


@dataclass
class SceneConfig:
    data: dict


@dataclass
class RenderConfig:
    width: int
    height: int
    hdri: str | None
    render_algorithm: str
    max_samples: int
    max_specular_depth: int
    denoise: bool
    output_path: Path


def load_configs() -> tuple[dict, dict]:
    if not SCENE_CONFIG_FILE.exists():
        raise FileNotFoundError("Scene config file not found")

    if not RENDER_CONFIG_FILE.exists():
        raise FileNotFoundError("Render config file not found")

    scene_config = SceneConfig(json.load(open(SCENE_CONFIG_FILE)))

    render_data = json.load(open(RENDER_CONFIG_FILE))
    render_data["output_path"] = Path(render_data["output_path"])
    render_config = RenderConfig(**render_data)

    return scene_config, render_config
