import numpy as np

from ray_tracer.application import Shader
from ray_tracer.domain import Shape
from ray_tracer.infrastructure.numpy.base import (
    FARAWAY,
    NumpyVector3D,
)


class NumpySphere(Shape):
    def __init__(
        self,
        center: NumpyVector3D,
        radius: float,
        shader: Shader,
    ) -> None:
        self.center = center
        self.position = center
        self.radius = radius
        self.shader = shader

    def intersect(self, ray_origin: NumpyVector3D, normalized_ray_direction: NumpyVector3D) -> float:
        """This function calculates the intersection between a ray (defined by its origin
        and direction) and a sphere (defined by self.position and self.radius).
        It returns the distance between the ray's origin and the intersection point,
        or a special value (FARAWAY) if no intersection exists.
        """
        projected_ray_direction = 2 * normalized_ray_direction.dot(ray_origin - self.position)
        ajusted_squared_distance = (
            abs(self.position) + abs(ray_origin) - 2 * self.position.dot(ray_origin) - (self.radius * self.radius)
        )
        discriminator = (projected_ray_direction**2) - (4 * ajusted_squared_distance)  # if < 0, no intersection
        discriminator_square_root = np.sqrt(np.maximum(0, discriminator))

        potential_solution_0 = (-projected_ray_direction - discriminator_square_root) / 2
        potential_solution_1 = (-projected_ray_direction + discriminator_square_root) / 2

        solution = np.where(
            (potential_solution_0 > 0) & (potential_solution_0 < potential_solution_1),
            potential_solution_0,
            potential_solution_1,
        )

        condition_to_have_intersection = (discriminator > 0) & (solution > 0)
        return np.where(condition_to_have_intersection, solution, FARAWAY)

    def get_uvs(self, hit_point: NumpyVector3D) -> tuple[np.ndarray, np.ndarray]:
        normals = hit_point - self.center

        u = 0.5 + np.arctan2(normals.z, normals.x) / (2 * np.pi)
        v = 0.5 - np.arcsin(normals.y) / np.pi

        return u, v


class NumpyTexturedSphere(NumpySphere):
    def __init__(
        self,
        center: NumpyVector3D,
        radius: float,
        shader: Shader,
    ) -> None:
        self.center = center
        self.position = center
        self.radius = radius
        self.shader = shader
