import numbers
from functools import reduce
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image

from ray_tracer.application import Renderer
from ray_tracer.domain import Camera, RGBColor, Scene3D, Vector3D

FARAWAY = 1.0e39


def extract(cond: ArrayLike, x: numbers.Number | np.ndarray) -> numbers.Number | np.ndarray:
    """Extracts elements from an array or returns the number itself if it is a scalar.

    Args:
        cond (array-like): Condition array to extract elements.
        x: Input number or array.
    """
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


class NumpyVector3D(Vector3D):
    """Represents a 3D vector with numpy-based operations."""

    def __init__(self, x: int | float, y: int | float, z: int | float) -> None:
        (self.x, self.y, self.z) = (x, y, z)

    def dot(self, other: Self) -> float:
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self) -> float:
        return self.dot(self)

    def components(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __mul__(self, other: int | float) -> Self:
        if isinstance(other, NumpyRGBColor | NumpyVector3D):
            return NumpyVector3D(self.x * other.x, self.y * other.y, self.z * other.z)
        return NumpyVector3D(self.x * other, self.y * other, self.z * other)

    def __add__(self, other: Self) -> Self:
        return NumpyVector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Self) -> Self:
        return NumpyVector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def norm(self) -> Any:
        """Normalizes the vector to have a magnitude of 1."""
        mag = np.sqrt(self.dot(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def extract(self, cond: ArrayLike) -> Self | "NumpyVectorArray3D":
        """Extracts components of the vector based on a condition."""
        if isinstance(cond, numbers.Number):
            return self

        return NumpyVectorArray3D(extract(cond, self.x), extract(cond, self.y), extract(cond, self.z))

    def place(self, cond: ArrayLike) -> Self:
        """Places the vector's components into a new vector."""
        r = NumpyVector3D(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r


class NumpyVectorArray3D(NumpyVector3D):
    pass


class NumpyRGBColor(NumpyVector3D):
    pass


class NumpyRenderer(Renderer):
    def raytrace_scene(
        self,
        ray_origin: NumpyVector3D,
        normalized_ray_direction: NumpyVector3D,
        scene: Scene3D,
    ) -> NumpyRGBColor:
        distances = [shape.intersect(ray_origin, normalized_ray_direction) for shape in scene.shapes]
        nearest_distance = reduce(np.minimum, distances)

        color = NumpyRGBColor(0, 0, 0)

        for shape, distance in zip(scene.shapes, distances, strict=False):
            hit: bool = (nearest_distance != FARAWAY) & (distance == nearest_distance)

            if np.any(hit):  # keep only the intersected points
                extracted_distance_to_intersection: np.ndarray = extract(hit, distance)
                extracted_ray_origin: NumpyVector3D = ray_origin.extract(hit)
                extracted_normalized_ray_direction: NumpyVectorArray3D = normalized_ray_direction.extract(hit)

                computed_color = shape.shader.create(
                    shape,
                    scene,
                    extracted_ray_origin,
                    extracted_normalized_ray_direction,
                    extracted_distance_to_intersection,
                    self,
                )

                color += computed_color.place(hit)

        return color

    def get_ray_directions(self, camera: Camera) -> np.ndarray:
        """1. Adjusts the aspect ratio of the camera to fit the image desired size.
        2. Creates a grid of points in the camera's view frustum (like pixels on a screen).
        3. Normalizes the grid points to create ray directions.
        4. Returns the normalized ray directions.
        The grid is created by tiling and repeating the x and y coordinates based on the camera's width and height.
        """
        aspect_ratio = float(camera.width) / camera.height
        screen = (-1, 1 / aspect_ratio + 0.25, 1, -1 / aspect_ratio + 0.25)
        x = np.tile(
            np.linspace(screen[0], screen[2], camera.width),
            camera.height,
        )
        y = np.repeat(
            np.linspace(screen[1], screen[3], camera.height),
            camera.width,
        )

        return (NumpyVector3D(x, y, 0) - camera.position).norm()

    def save_image(self, color: RGBColor, camera: Camera, output_path: str) -> None:
        rgb_colors = [
            Image.fromarray(
                (255 * np.clip(layer, 0, 1).reshape((camera.height, camera.width))).astype(np.uint8),
                "L",
            )
            for layer in color.components()
        ]
        Image.merge("RGB", rgb_colors).save(output_path)
