import time
from pathlib import Path

from ray_tracer.domain import RenderConfig, Scene
from ray_tracer.infrastructure.numpy import RayTracer
from ray_tracer.infrastructure.pillow import PillowImageService
from ray_tracer.services import render_single_image_pipeline

if __name__ == "__main__":
    ray_tracer = RayTracer()
    image_service = PillowImageService()

    scene = Scene.from_csv(Path("tests/testdata/input_scene.csv"))
    render_config = RenderConfig.from_json(
        Path("tests/testdata/input_render_settings.json")
    )

    start = time.time()
    if render_single_image_pipeline(scene, render_config, ray_tracer, image_service):
        print(f"Rendered image with success in {time.time() - start:.2f} seconds")
