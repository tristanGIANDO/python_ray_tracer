from pathlib import Path
from typing import Any

from PIL import Image

from old_ray_tracer.services import ImageService


class PillowImageService(ImageService):
    def from_path(self, image_path: Path) -> Image:
        return Image.open(image_path)

    def to_bitmap(self, image: Any, output_path: Path) -> None:
        Image.fromarray(image).save(output_path)
