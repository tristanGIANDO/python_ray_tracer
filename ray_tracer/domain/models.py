from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ray_tracer.domain.vector import Vector3D

FARAWAY = 1.0e39


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
class Light:
    intensity: float


@dataclass
class PointLight(Light):  # TODO: add intensity
    """A pointLight is a point that emits rays in all directions and is used as a light source."""

    position: Vector3D


@dataclass
class DomeLight(Light):
    """The dome light (or sky light) is an omnidirectional light source that simulates the ambient lighting
    of an environment.
    """

    color: Vector3D


class Shape(ABC):
    @abstractmethod
    def intersect(self, ray_origin: Vector3D, normalized_ray_direction: Vector3D) -> None:
        pass


@dataclass
class Scene3D:
    """The 3D scene groups together all the 3D elements that will interact with each other."""

    shapes: list[Shape]
    lights: list[PointLight | DomeLight]
    camera: Camera


@dataclass
class Slot:
    color: Vector3D | Path | None
    intensity: float


@dataclass
class Specular(Slot):
    roughness: float


class Diffuse(Slot):
    pass
