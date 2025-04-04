from abc import ABC, abstractmethod
from dataclasses import dataclass


class Vector3D:
    pass


class RGBColor(Vector3D):
    pass


@dataclass
class Camera:
    position: Vector3D
    width: int
    height: int


@dataclass
class PointLight:  # TODO: add intensity
    position: Vector3D


class Shape(ABC):
    @abstractmethod
    def intersect(self, ray_origin: Vector3D, normalized_ray_direction: Vector3D):
        pass

    @abstractmethod
    def diffusecolor(self, intersection_point: Vector3D) -> RGBColor:
        pass


@dataclass
class Scene3D:
    shapes: list[Shape]
    lights: list[PointLight]
    camera: Camera
