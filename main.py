import time
from pathlib import Path

from ray_tracer.application import render_image_pipeline
from ray_tracer.domain import Camera, PointLight, Scene3D
from ray_tracer.infrastructure import (
    CheckeredSphere,
    NumpyRenderer,
    NumpyRGBColor,
    NumpySphere,
    NumpyVector3D,
)

if __name__ == "__main__":
    renderer = NumpyRenderer(0.0, 1.0, 0.5)

    scene = Scene3D(
        [
            NumpySphere(
                NumpyVector3D(0.55, 0.5, 3),
                1.0,
                NumpyRGBColor(0, 1, 1),
            ),
            NumpySphere(
                NumpyVector3D(-0.45, 0.1, 1), 0.4, NumpyRGBColor(0.5, 0.5, 0.5)
            ),
            CheckeredSphere(
                NumpyVector3D(0, -99999.5, 0),
                99999,
                NumpyRGBColor(0.18, 0.18, 0.18),
                1.0,
            ),
        ],
        [PointLight(NumpyVector3D(5, 10, -10))],
        Camera(NumpyVector3D(0, 0.35, -2), 400, 300),
    )

    output_path = Path("render.png")

    start_time = time.time()
    render_image_pipeline(scene, output_path, renderer)
    print("Took", time.time() - start_time)
