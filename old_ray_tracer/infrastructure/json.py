import json
from pathlib import Path

from old_ray_tracer.domain import Light, RenderConfig, Scene, Sphere, Vector3D


def scene_from_json(file_path: Path) -> Scene:
    # open json file and load data
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    objects = []
    lights = []
    for object in data:
        if object["type"] == "Sphere":
            objects.append(
                Sphere(
                    Vector3D(*object["centerXYZ"]),
                    object["radius"],
                    Vector3D(*object["colorRGB"]),
                    object["reflection"],
                    object["roughness"],
                    object["texture"],
                )
            )
        elif object["type"] == "Light":
            lights.append(
                Light(
                    Vector3D(*object["centerXYZ"]),
                    Vector3D(*object["intensityRGB"]),
                )
            )

    return Scene(objects, lights)


def render_config_from_json(file_path: Path) -> RenderConfig:
    data = json.load(file_path.open())
    data["output_path"] = Path(data["output_path"])
    data["background"] = Path(data["background"])

    return RenderConfig(**data)
