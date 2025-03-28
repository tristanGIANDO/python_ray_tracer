import time
from pathlib import Path

from ray_tracer.infrastructure.json import render_config_from_json, scene_from_json
from ray_tracer.infrastructure.numpy import NumpyRayTracer
from ray_tracer.infrastructure.pillow import PillowImageService
from ray_tracer.services import render_single_image_pipeline

if __name__ == "__main__":
    image_service = PillowImageService()

    ray_tracer = NumpyRayTracer(image_service)

    scene = scene_from_json(Path("tests/testdata/input_scene.json"))
    render_config = render_config_from_json(
        Path("tests/testdata/input_render_settings.json")
    )

    start = time.time()
    if render_single_image_pipeline(scene, render_config, ray_tracer, image_service):
        print(f"Rendered image with success in {time.time() - start:.2f} seconds")
