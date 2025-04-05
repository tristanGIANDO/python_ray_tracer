from abc import ABC, abstractmethod
from dataclasses import dataclass


class Vector3D:
    def __init__(self, x: float, y: float, z: float) -> None:
        (self.x, self.y, self.z) = (x, y, z)


class RGBColor(Vector3D):
    pass


@dataclass
class Camera:
    """This object represents the observation point.
    In a context of transformation from 3D to 2D, the term Camera seems appropriate
    to include notions of field of view size.
    """

    position: Vector3D
    width: int
    height: int


@dataclass
class PointLight:  # TODO: add intensity
    """A pointLight is a point that emits rays in all directions and is used
    as a light source."""

    position: Vector3D


@dataclass
class DomeLight:
    """The dome light (or sky light) is an omnidirectional light source that simulates
    the ambient lighting of an environment."""

    intensity: float
    color: RGBColor


class Shape(ABC):
    @abstractmethod
    def intersect(self, ray_origin: Vector3D, normalized_ray_direction: Vector3D):
        pass

    @abstractmethod
    def diffusecolor(self, intersection_point: Vector3D) -> RGBColor:
        pass


@dataclass
class Scene3D:
    """The 3D scene groups together all the 3D elements that will interact with each other."""

    shapes: list[Shape]
    lights: list[PointLight]
    camera: Camera
