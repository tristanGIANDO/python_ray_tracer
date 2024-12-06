import numpy as np
from PIL import Image

from src.objects import Sphere, Light
from src.configs import SceneConfig, RenderConfig
from src.ray_tracing import render
from src.vectors import Vector3D


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
