import time
from pathlib import Path

from ray_tracer.application import render_image_pipeline
from ray_tracer.domain import Camera, DomeLight, PointLight, Scene3D
from ray_tracer.infrastructure import (
    NumpyRenderer,
    NumpyRGBColor,
    NumpyShader,
    NumpySphere,
    NumpyVector3D,
    Texture,
    TextureChecker,
)

if __name__ == "__main__":
    renderer = NumpyRenderer()

    scene = Scene3D(
        [
            NumpySphere(
                NumpyVector3D(0.55, 0.5, 3),
                1.0,
                NumpyShader(0.1, 1.0, 0.5, 0.1, 1.0, TextureChecker()),
            ),
            NumpySphere(
                NumpyVector3D(-0.45, 0.1, 1),
                0.4,
                NumpyShader(0.0, 0.05, 0.5, 0.0, 1.0, Texture(NumpyRGBColor(1, 0, 0))),
            ),
            NumpySphere(
                NumpyVector3D(0, -99999.5, 0),
                99999,
                NumpyShader(1.0, 0.0, 0.0, 0.0, 1.0, TextureChecker()),
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
