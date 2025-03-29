import time
from pathlib import Path

import numpy as np
from PIL import Image

from ray_tracer.domain import (
    Camera,
    PointLight,
    RenderImage,
    RGBColor,
    Scene3D,
    Vector3D,
)
from ray_tracer.services import RayTracer


def main(
    render_image: RenderImage,
    camera: Camera,
    scene: Scene3D,
    output_path: Path,
    ray_tracer: RayTracer,
    reflection_gain: float,
    specular_gain: float,
) -> None:
    aspect_ratio = float(render_image.width) / render_image.height
    projection_canva = (-1, 1 / aspect_ratio + 0.25, 1, -1 / aspect_ratio + 0.25)
    x = np.tile(
        np.linspace(projection_canva[0], projection_canva[2], render_image.width),
        render_image.height,
    )
    y = np.repeat(
        np.linspace(projection_canva[1], projection_canva[3], render_image.height),
        render_image.width,
    )

    start_time = time.time()
    ray_destinations = Vector3D(x, y, 0)
    color = ray_tracer.render(
        camera,
        light,
        camera.position,
        (ray_destinations - camera.position).norm(),
        scene,
        reflection_gain,
        specular_gain,
    )
    print("Took", time.time() - start_time)

    rgb_colors = [
        Image.fromarray(
            (
                255
                * np.clip(grey_color, 0, 1).reshape(
                    (render_image.height, render_image.width)
                )
            ).astype(np.uint8),
            "L",
        )
        for grey_color in color.components()
    ]
    Image.merge("RGB", rgb_colors).save(output_path)


if __name__ == "__main__":
    from ray_tracer.infra_numpy import CheckeredSphere, NumpyRayTracer, NumpySphere

    light = PointLight(Vector3D(5, 10, -10))
    camera = Camera(Vector3D(0, 0.35, -2))
    render_image = RenderImage(4000, 3000)
    ray_tracer = NumpyRayTracer()

    scene = Scene3D(
        [
            NumpySphere(Vector3D(0.55, 0.5, 3), 1.0, RGBColor(0, 1, 1)),
            NumpySphere(Vector3D(-0.45, 0.1, 1), 0.4, RGBColor(0.5, 0.5, 0.5)),
            CheckeredSphere(
                Vector3D(0, -99999.5, 0), 99999, RGBColor(0.18, 0.18, 0.18), 1.0
            ),
        ]
    )

    output_path = Path("render.png")

    main(render_image, camera, scene, output_path, ray_tracer, 0.5, 1.0)
