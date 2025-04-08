import time
from pathlib import Path

from ray_tracer.application import render_image_pipeline
from ray_tracer.domain import Camera, DomeLight, PointLight, Scene3D
from ray_tracer.infrastructure import (
    CheckeredSphere,
    NumpyRenderer,
    NumpyRGBColor,
    NumpySphere,
    NumpyVector3D,
)

if __name__ == "__main__":
    renderer = NumpyRenderer(
        reflection_gain=0.8,
        specular_gain=1.0,
        specular_roughness=0.5,
        iridescence_gain=0.05,
        diffuse_gain=1.0,
    )

    scene = Scene3D(
        [
            NumpySphere(
                NumpyVector3D(0.55, 0.5, 3),
                1.0,
                NumpyRGBColor(1, 0, 1),
            ),
            NumpySphere(NumpyVector3D(-0.45, 0.1, 1), 0.4, NumpyRGBColor(0.5, 0.5, 0.5)),
            CheckeredSphere(
                NumpyVector3D(0, -99999.5, 0),
                99999,
                NumpyRGBColor(
                    1,
                    1,
                    1,
                ),
                1.0,
            ),
        ],
        [
            PointLight(NumpyVector3D(5, 10, -10)),
            DomeLight(0.1, NumpyRGBColor(1, 1, 1)),
        ],
        Camera(NumpyVector3D(0, 0.2, -2), int(1920 / 2), int(1080 / 2)),
    )

    output_path = Path("render.png")

    start_time = time.time()
    render_image_pipeline(scene, output_path, renderer)
    print("Took", time.time() - start_time)
