import time

from configs.configs import load_configs
from ray_tracer.scenes import build_scene, render_single_image


def main():
    start = time.time()
    scene_config, render_config = load_configs()
    print(f"Loaded configs in {time.time() - start:.2f} seconds")
    start = time.time()
    scene_content = build_scene(scene_config)
    print(f"Built scene in {time.time() - start:.2f} seconds")
    start = time.time()
    render_single_image(scene_content, render_config, log_results=True)
    print(f"Rendered images in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
