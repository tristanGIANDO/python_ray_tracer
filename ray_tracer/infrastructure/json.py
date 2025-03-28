import json
from pathlib import Path

from ray_tracer.domain import Light, RenderConfig, Scene, Sphere


def scene_from_json(file_path: Path) -> Scene:
    object_data = json.load(file_path.open())
    objects = []
    lights = []
    for object in object_data:
        if object["type"] == "Sphere":
            objects.append(
                Sphere(
                    object["center"],
                    object["radius"],
                    object["color"],
                    object["reflection"],
                    object["roughness"],
                    object["texture"],
                )
            )
        elif object["type"] == "Light":
            lights.append(
                Light(
                    object["position"],
                    object["intensity"],
                )
            )

    return Scene(objects, lights)


def render_config_from_json(file_path: Path) -> RenderConfig:
    data = json.load(file_path.open())
    return RenderConfig(**data)
