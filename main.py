import time
from pathlib import Path

import numpy as np
from PIL import Image

from ray_tracer.application import RenderService
from ray_tracer.domain import (
    Camera,
    PointLight,
    RenderImage,
    Scene3D,
)
from ray_tracer.infra_numpy import get_rays_destinations


def main(
    render_image: RenderImage,
    camera: Camera,
    scene: Scene3D,
    output_path: Path,
    render_service: RenderService,
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

    normalized_ray_destinations = get_rays_destinations(render_image, camera)

    color = render_service.render_scene(
        camera,
        light,
        camera.position,
        normalized_ray_destinations,
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
    from ray_tracer.infra_numpy import (
        CheckeredSphere,
        NumpyRenderService,
        NumpyRGBColor,
        NumpySphere,
        NumpyVector3D,
    )

    light = PointLight(NumpyVector3D(5, 10, -10))
    camera = Camera(NumpyVector3D(0, 0.35, -2))
    render_image = RenderImage(400, 300)
    render_service = NumpyRenderService()

    scene = Scene3D(
        [
            NumpySphere(NumpyVector3D(0.55, 0.5, 3), 1.0, NumpyRGBColor(0, 1, 1)),
            NumpySphere(
                NumpyVector3D(-0.45, 0.1, 1), 0.4, NumpyRGBColor(0.5, 0.5, 0.5)
            ),
            CheckeredSphere(
                NumpyVector3D(0, -99999.5, 0),
                99999,
                NumpyRGBColor(0.18, 0.18, 0.18),
                1.0,
            ),
        ]
    )

    output_path = Path("render.png")

    main(render_image, camera, scene, output_path, render_service, 0.5, 1.0)
