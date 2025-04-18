from dataclasses import dataclass
from pathlib import Path

from ray_tracer.domain.vector import Vector3D

FARAWAY = 1.0e39


@dataclass
class Shape:
    def intersect(self, ray_origin: Vector3D, normalized_ray_direction: Vector3D) -> None:
        raise NotImplementedError

    def get_normals(self, hit_point: Vector3D) -> Vector3D:
        """Returns the normal vector of the shape at the intersection point."""
        raise NotImplementedError

    def get_uvs(self, hit_point: Vector3D) -> tuple[float, float]:
        """Returns the UV coordinates of the shape at the intersection point."""
        raise NotImplementedError


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
    position: Vector3D


@dataclass
class PointLight(Light):  # TODO: add intensity
    """A pointLight is a point that emits rays in all directions and is used as a light source."""

    pass


@dataclass
class DomeLight(Light):
    """The dome light (or sky light) is an omnidirectional light source that simulates the ambient lighting
    of an environment.
    """

    color: Vector3D


@dataclass
class Slot:
    color: Vector3D | Path | None
    intensity: float


@dataclass
class Specular(Slot):
    roughness: float
    ior: float


class Diffuse(Slot):
    pass


@dataclass
class Shader:
    diffuse: Diffuse
    specular: Specular


@dataclass
class Scene3D:
    """The 3D scene groups together all the 3D elements that will interact with each other."""

    shapes: list[tuple[Shape, Shader]]
    lights: list[PointLight | DomeLight]
    camera: Camera
