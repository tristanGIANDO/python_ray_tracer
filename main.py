import time
from pathlib import Path

from configs.configs import load_render_settings
from ray_tracer.scenes import load_scene, render_single_image


def main(scene_csv_file: Path, render_settings_json_file: Path) -> None:
    render_config = load_render_settings(render_settings_json_file)
    scene_content = load_scene(scene_csv_file)
    start = time.time()
    render_single_image(scene_content, render_config, log_results=True)
    print(f"Rendered images in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main(
        Path("tests/testdata/input_scene.csv"),
        Path("tests/testdata/input_render_settings.json"),
    )
