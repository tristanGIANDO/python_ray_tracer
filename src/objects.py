import numpy as np

from src.vectors import Vector3D
from src.utils import get_texture_color


class Light:
    def __init__(self, position: Vector3D, intensity: Vector3D) -> None:
        """
        Initializes a light source.

        Args:
            position (Vector3D): The position of the light source.
            intensity (Vector3D): The intensity and color of the light.

        The position determines where the light is located in the scene, and the intensity determines how bright the light is and its color.
        """
        self.position = position
        self.intensity = intensity


class Sphere:
    def __init__(
        self,
        center: Vector3D,
        radius: Vector3D,
        color: Vector3D,
        reflection: float | None = 0.5,
        texture=None,
    ) -> None:
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
        self.center = center
        self.radius = radius
        self.color = color
        self.reflection = reflection
        self.texture = texture

    def intersect(self, ray_origin: Vector3D, ray_dir: Vector3D) -> float | None:
        """
        Computes the intersection of a ray with the sphere.

        Args:
            ray_origin (Vector3D): The origin of the ray.
            ray_dir (Vector3D): The direction of the ray.

        Returns:
            float or None: The distance from the ray origin to the intersection point, or None if no intersection.

        This method uses the mathematical formula for the intersection of a ray and a sphere.
        It solves a quadratic equation to find where (if at all) the ray hits the sphere.
        The discriminant (`b^2 - 4 * c`) determines if there are intersections (real solutions).
        If the discriminant is positive, there are two possible intersection points.
        """
        oc = ray_origin - self.center
        b = 2.0 * oc.dot(ray_dir)
        c = oc.dot(oc) - self.radius**2
        disc = b**2 - 4 * c
        if disc > 0:
            sqrtd = np.sqrt(disc)
            t0 = (-b - sqrtd) / 2
            t1 = (-b + sqrtd) / 2
            if t0 > 0:
                return t0
            if t1 > 0:
                return t1
        return None

    def get_surface_color(self, hit_point: Vector3D) -> Vector3D:
        """
        Gets the color of the surface at a given hit point.

        Args:
            hit_point (Vector3D): The point on the surface of the sphere.

        Returns:
            Vector3D: The color at the hit point.

        If the sphere has a texture, this method calculates texture coordinates (u, v) based on the position of the hit point.
        It then uses these coordinates to look up the color from the texture image. If no texture is present, it returns the base color.
        """
        try:
            if self.texture.any():
                normal = (hit_point - self.center).norm()
                u = 0.5 + np.arctan2(normal.z, normal.x) / (2 * np.pi)
                v = 0.5 - np.arcsin(normal.y) / np.pi
                return get_texture_color(self.texture, u, v)
        except AttributeError:
            return self.color
