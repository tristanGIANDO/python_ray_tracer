import numpy as np
from PIL import Image
import time
import json
from src.vectors import Vector3D
from src.objects import Sphere, Light
from src.ray_tracing import render
from dataclasses import dataclass


@dataclass
class SceneConfig:
    data: dict


@dataclass
class RenderConfig:
    data: dict


def load_configs() -> tuple[dict, dict]:
    scene_settings = SceneConfig(json.load(open("configs/scene_settings.json")))
    render_settings = RenderConfig(json.load(open("configs/render_settings.json")))

    return scene_settings, render_settings


def build_scene(config: SceneConfig) -> tuple[list[Sphere], list[Light]]:
    objects = []
    lights = []

    for settings in config.data.values():
        if settings["type"] == "Light":
            lights.append(
                Light(Vector3D(*settings["position"]), Vector3D(*settings["intensity"]))
            )

        elif settings["type"] == "Sphere":
            texture = (
                np.array(Image.open(settings["texture"]))
                if "texture" in settings
                else None
            )
            objects.append(
                Sphere(
                    center=Vector3D(*settings["center"]),
                    radius=settings["radius"],
                    color=Vector3D(*settings["color"]),
                    reflection=settings["reflection"],
                    texture=texture,
                )
            )

    return objects, lights


def batch_render(scene_content, config: RenderConfig):
    objects, lights = scene_content

    for settings in list(config.data.values()):
        image = render(objects, lights, **settings)
        Image.fromarray((image * 255).astype(np.uint8)).save("render.png")


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
