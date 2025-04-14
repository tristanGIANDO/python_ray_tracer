import time
from pathlib import Path

from ray_tracer.application import render_image_pipeline
from ray_tracer.domain import Camera, DomeLight, PointLight, Scene3D
from ray_tracer.infrastructure.numpy.base import NumpyRenderer, NumpyVector3D
from ray_tracer.infrastructure.numpy.shader import NumpyShader, Texture, TextureChecker
from ray_tracer.infrastructure.numpy.shape import NumpyRGBColor, NumpySphere, NumpyTexturedSphere

if __name__ == "__main__":
    renderer = NumpyRenderer()

    scene = Scene3D(
        [
            NumpySphere(
                NumpyVector3D(0.55, 0.5, 3),
                0.5,
                NumpyShader(
                    reflection_gain=0.0,
                    specular_gain=0,
                    specular_roughness=0.01,
                    iridescence_gain=0,
                    diffuse_gain=0.0,
                    diffuse_color=Texture(NumpyRGBColor(1, 1, 1)),
                ),
            ),
            NumpyTexturedSphere(
                NumpyVector3D(-0.45, 0.1, 1),
                1,
                NumpyShader(
                    reflection_gain=0,
                    specular_gain=0.1,
                    specular_roughness=0.1,
                    iridescence_gain=0.0,
                    diffuse_gain=1.0,
                    diffuse_color=Texture(NumpyRGBColor(1, 1, 1)),
                ),
                Path("sourceimages/2k_mars.jpg"),
            ),
            NumpySphere(
                NumpyVector3D(0, -99999.5, 0),
                99999,
                NumpyShader(0.0, 0.1, 0.5, 0.0, 1.0, TextureChecker()),
            ),
        ],
        [  # TODO: use multiple lights
            PointLight(NumpyVector3D(-5, 5, -10)),
            # PointLight(NumpyVector3D(-2, 1, 2)),
            DomeLight(0.1, NumpyRGBColor(1, 1, 1)),
        ],
        Camera(NumpyVector3D(0, 0.2, -2), int(1920 / 2), int(1080 / 2)),
    )

    output_path = Path("render.png")

    start_time = time.time()
    render_image_pipeline(scene, output_path, renderer)
    print("Took", time.time() - start_time)  # noqa: T201
