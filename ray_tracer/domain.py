import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def extract(cond, x) -> numbers.Number | np.ndarray:  # TODO: where to put this?
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


class Vector3D:
    def __init__(self, x: int | float, y: int | float, z: int | float) -> None:
        (self.x, self.y, self.z) = (x, y, z)

    def __mul__(self, other) -> "Vector3D":
        return Vector3D(self.x * other, self.y * other, self.z * other)

    def __add__(self, other) -> "Vector3D":
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other) -> "Vector3D":
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self) -> int | float:
        return self.dot(self)

    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def components(self) -> tuple[int | float, int | float, int | float]:
        return (self.x, self.y, self.z)

    def extract(self, cond) -> "Vector3D":
        return Vector3D(
            extract(cond, self.x), extract(cond, self.y), extract(cond, self.z)
        )

    def place(self, cond) -> "Vector3D":
        r = Vector3D(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r


RGBColor = Vector3D


@dataclass
class Camera:
    position: Vector3D


@dataclass
class PointLight:  # TODO: add intensity
    position: Vector3D


class Shape(ABC):
    @abstractmethod
    def intersect(self, ray_origin: Vector3D, normalized_ray_direction: Vector3D):
        pass

    @abstractmethod
    def diffusecolor(self, intersection_point: Vector3D):
        pass

    @abstractmethod
    def create_shader(  # TODO: sortir de la classe Shape
        self,
        ray_origin: Vector3D,
        normalized_ray_direction: Vector3D,
        distance_origin_to_intersection,
        scene,  # TODO: move this
        ray_tracer,  # TODO: move this
        camera: Camera,
        light: PointLight,
        reflection_gain: float,
        specular_gain: float,
    ):
        pass


@dataclass
class RenderImage:  # TODO: transformer en render_config
    width: int
    height: int


@dataclass
class Scene3D:  # TODO: ajouter lights
    shapes: list[Shape]
