import time

from src.configs import load_configs
from src.scenes import build_scene, batch_render


def main():
    start = time.time()
    scene_config, render_config = load_configs()
    print(f"Loaded configs in {time.time() - start:.2f} seconds")
    start = time.time()
    scene_content = build_scene(scene_config)
    print(f"Built scene in {time.time() - start:.2f} seconds")
    start = time.time()
    batch_render(scene_content, render_config)
    print(f"Rendered images in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
