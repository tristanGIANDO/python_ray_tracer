import time
from pathlib import Path

from ray_tracer.application import render_image_pipeline
from ray_tracer.domain.models import (
    Camera,
    Diffuse,
    DomeLight,
    PointLight,
    Scene3D,
    Shader,
    Specular,
)
from ray_tracer.domain.vector import Vector3D
from ray_tracer.infrastructure import NumpyRenderer, Sphere3D

if __name__ == "__main__":
    scene = Scene3D(
        [
            (
                Sphere3D(Vector3D(0.55, 0.5, 3), 0.5),
                Shader(
                    Diffuse(Path("sourceimages/hdri_2.jpg"), 0.5),
                    Specular(Vector3D(0, 1, 1), 0.5, 0.5, 1.5),
                ),
            ),
            (
                Sphere3D(Vector3D(-0.55, 0.1, 1), 0.5),
                Shader(Diffuse(Path("sourceimages/2k_mars.jpg"), 1.0), Specular(Vector3D(1, 1, 1), 1.0, 0.9, 1.5)),
            ),
            (
                Sphere3D(Vector3D(0, -99999.5, 0), 99999),
                Shader(
                    Diffuse(Vector3D(1, 0, 1), 0.1),
                    Specular(Vector3D(0, 1, 1), 0.5, 0.5, 1.5),
                ),
            ),
        ],
        [  # TODO: use multiple lights
            PointLight(1.0, Vector3D(-5, 5, -10)),
            DomeLight(0.1, Vector3D(0.0, 0.0, 0.0), Vector3D(1, 1, 1)),
        ],
        Camera(Vector3D(0, 0.2, -2), int(1920 / 2), int(1080 / 2)),
    )

    start_time = time.time()
    render_image_pipeline(scene, Path("render.png"), NumpyRenderer())
    print("Took", time.time() - start_time)  # noqa: T201
