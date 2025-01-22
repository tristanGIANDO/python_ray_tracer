import json
from dataclasses import dataclass
from pathlib import Path


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


def load_render_settings(render_settings_json_file: Path) -> dict:
    if not render_settings_json_file.exists():
        raise FileNotFoundError("Render config file not found")

    render_data = json.load(open(render_settings_json_file))
    render_data["output_path"] = Path(render_data["output_path"])
    render_config = RenderConfig(**render_data)

    return render_config
