import numpy as np
from PIL import Image
import time
import json
from pathlib import Path
from src.vector import Vec3
from src.objects import Sphere
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


render_settings = json.load(open("configs/render_settings.json"))

for test_index, settings in render_settings.items():
    start = time.time()
    image = render(scene, **settings)
    print(f"Rendered {test_index} in {time.time() - start:.2f} seconds")

    output_image_filepath = Path("output") / f"{test_index}_render.png"
    Image.fromarray((image * 255).astype(np.uint8)).save(output_image_filepath)
