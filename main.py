import numpy as np
from PIL import Image
import time
import json
from pathlib import Path
from src.vector import Vec3
from src.objects import Sphere, Light, EmissiveSphere
from src.path_tracing import render

scene_settings = json.load(open("configs/scene_settings.json"))

scene = []
for settings in scene_settings.values():
    texture = Image.open(settings["texture"]) if "texture" in settings else None
    scene.append(
        Sphere(
            Vec3(*settings["center"]),
            settings["radius"],
            Vec3(*settings["color"]),
            settings["reflection"],
            texture,
        )
    )

lights = [
    Light(Vec3(5, 5, -10), Vec3(1, 1, 1)),  # Lumière ponctuelle
    EmissiveSphere(Vec3(0, 5, 3), 1, Vec3(1, 0.8, 0.6), 5)  # Lumière de surface
]


render_settings = json.load(open("configs/render_settings.json"))

for test_index, settings in render_settings.items():
    start = time.time()
    image = render(scene, **settings)
    print(f"Rendered {test_index} in {time.time() - start:.2f} seconds")

    output_image_filepath = Path("output") / f"{test_index}_render.png"
    Image.fromarray((image * 255).astype(np.uint8)).save(output_image_filepath)
