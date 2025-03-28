from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


@dataclass
class Vector3D(BaseModel):
    """
    Initializes a 3D vector.

    Args:
        x (float): X component of the vector.
        y (float): Y component of the vector.
        z (float): Z component of the vector.

    This class is used to represent a point or direction in 3D space. Each vector has three components: x, y, and z.
    """

    x: float
    y: float
    z: float


@dataclass
class Light(BaseModel):
    """
    Initializes a light source.

    Args:
        position (Vector3D): The position of the light source.
        intensity (Vector3D): The intensity and color of the light.

    The position determines where the light is located in the scene, and the intensity determines how bright the light is and its color.
    """

    position: Vector3D
    intensity: Vector3D


@dataclass
class Sphere(BaseModel):
    """
    Initializes a sphere object.

    Args:
        center (Vector3D): The center position of the sphere.
        radius (float): The radius of the sphere.
        color (Vector3D): The base color of the sphere.
        reflection (float, optional): Reflection coefficient of the sphere. Defaults to 0.5.
        texture (Image, optional): Texture image for the sphere. Defaults to None.

    A sphere is defined by its position, size, color, and optionally, a texture and reflection properties. The texture
    allows the sphere to have more complex surface details.
    """

    center: Vector3D
    radius: float
    color: Vector3D
    reflection: float | None
    roughness: float
    texture: str | None  # image ?


HDRI = Path


@dataclass
class Scene(BaseModel):
    """
    Initializes a scene with objects and lights.

    Args:
        objects (list[Sphere]): A list of sphere objects in the scene.
        lights (list[Light]): A list of light sources in the scene.
        environment (HDRI, optional): The high dynamic range image for the environment lighting. Defaults to None.

    A scene consists of a list of objects (spheres) and light sources. The environment can be set to provide background lighting.
    """

    objects: list[Sphere]
    lights: list[Light]
    background: Path


@dataclass
class RenderConfig(BaseModel):
    image_width: int
    image_height: int
    samples_per_pixel: int
    output_path: Path
