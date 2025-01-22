import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from configs.configs import RenderConfig, SceneConfig
from denoiser import denoise
from evaluation.utils import create_csv_file, populate_csv_file
from ray_tracer.objects import Light, Sphere
from ray_tracer.ray_tracing import render_monte_carlo_live
from ray_tracer.utils import HDRIEnvironment
from ray_tracer.vectors import Vector3D


def build_scene(render_config: SceneConfig) -> tuple[list[Sphere], list[Light]]:
    objects = []
    lights = []

    for settings in render_config.data.values():
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
                    roughness=settings["roughness"],
                    texture=texture,
                )
            )

    return objects, lights


def render_single_image(scene_content, render_config: RenderConfig, log_results: bool):
    objects, lights = scene_content

    if log_results:
        render_times_file = Path("dataset/render_times.csv")
        render_times_file.parent.mkdir(parents=True, exist_ok=True)
        columns = (
            ["image_name"]
            + list(asdict(render_config).keys())
            + ["render_time", "denoise_time", "path"]
        )
        create_csv_file(render_times_file, columns=columns)

    if log_results:
        start_time = time.time()

    environment_image = (
        HDRIEnvironment(render_config.hdri) if render_config.hdri else None
    )

    if render_config.render_algorithm == "monte_carlo":
        for partial_image in render_monte_carlo_live(
            objects,
            lights,
            render_config.width,
            render_config.height,
            environment_image,
            render_config.max_samples,
        ):
            plt.imshow(partial_image)
            plt.pause(0.01)

        image = partial_image

        if log_results:
            render_elapsed_time = time.time() - start_time
    else:
        raise ValueError("Invalid render algorithm")

    if render_config.denoise:
        if log_results:
            start_time = time.time()
        image = denoise(image)
        if log_results:
            denoise_elapsed_time = time.time() - start_time
    else:
        denoise_elapsed_time = 0

    output_path = render_config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image * 255).astype(np.uint8)).save(output_path)

    if log_results:
        populate_csv_file(
            render_times_file,
            [output_path.stem]
            + list(asdict(render_config).values())
            + [render_elapsed_time, denoise_elapsed_time, output_path],
        )
