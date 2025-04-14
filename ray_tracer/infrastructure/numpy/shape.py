from pathlib import Path

import numpy as np
from PIL import Image

from ray_tracer.application import Shader
from ray_tracer.domain import Shape
from ray_tracer.infrastructure.numpy.base import (
    FARAWAY,
    NumpyRGBColor,
    NumpyVector3D,
    NumpyVectorArray3D,
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

    def diffusecolor(self, intersection_point: NumpyVectorArray3D) -> NumpyRGBColor:
        return NumpyRGBColor(1, 1, 1)


class NumpyTexturedSphere(NumpySphere):
    def __init__(
        self,
        center: NumpyVector3D,
        radius: float,
        shader: Shader,
        texture_path: Path,  # TODO: move in shader
    ) -> None:
        self.center = center
        self.position = center
        self.radius = radius
        self.shader = shader

        self.texture = Image.open(texture_path).convert("RGB")
        # self.texture = np.asarray(image) / 255.0

    # def diffusecolor(self, intersection_point: NumpyVectorArray3D) -> NumpyRGBColor:
    #     normal = (intersection_point - self.center).norm()

    #     u = 0.5 + np.arctan2(normal.z, normal.x) / (2 * np.pi)
    #     v = 0.5 - np.arcsin(normal.y) / np.pi

    #     u = u % 1  # Repeat if u > 1
    #     v = v % 1
    #     height, width, _ = self.texture.shape

    #     i = (u * (width - 1)).astype(int) if isinstance(u, np.ndarray) else int(u * (width - 1))
    #     j = (v * (height - 1)).astype(int) if isinstance(v, np.ndarray) else int(v * (height - 1))

    #     color = self.texture[j, i, :3]
    #     if len(color.shape) > 1 and color.shape[0] > 1:
    #         color = color.mean(axis=0)  # Reduce to a single RGB value by averaging
    #     if color.shape != (3,):
    #         if color.shape == (1, 3):
    #             color = color.flatten()
    #         else:
    #             raise ValueError(f"Expected 3 color channels, but got {color.shape}")
    #     r, g, b = color
    #     return NumpyRGBColor(r, g, b)

    def diffusecolor(self, hit_point: NumpyVector3D) -> NumpyRGBColor:
        """Calcule la couleur diffusée à partir d'un point d'impact, en tenant compte de la texture.
        On considère que les composantes de hit_point et self.center sont des tableaux NumPy.
        """
        if self.texture:
            # Calculer le vecteur normal pour chaque hit_point.
            normals = hit_point - self.center
            # Calculer la norme sur l'axe approprié selon la structure de vos vecteurs.
            # Ici, j'assume que les données sont rangées de sorte à ce que l'axe 0
            # représente les composantes x, y, z par exemple.
            # norm = normals.norm()
            # normals = normals / norm

            # Calculer les coordonnées de texture (u, v) pour tous les points
            u = 0.5 + np.arctan2(normals.z, normals.x) / (2 * np.pi)
            v = 0.5 - np.arcsin(normals.y) / np.pi

            return get_texture_color(self.texture, u, v)
        else:
            return NumpyRGBColor(0.5, 0.5, 0.5)


def get_texture_color(texture: Image, u, v) -> NumpyRGBColor:
    """Récupère, de façon vectorisée, la couleur à partir d'une texture pour des coordonnées UV données.
    Les coordonnées u et v sont supposées être des tableaux NumPy.
    """
    # Convertir l'image en tableau NumPy (par exemple, shape = (hauteur, largeur, canaux)).
    texture_data = np.array(texture)

    # Gérer la répétition si u ou v sont supérieures à 1
    u = np.mod(u, 1)
    v = np.mod(v, 1)

    # Calculer les indices dans l'image
    i = (u * (texture_data.shape[1] - 1)).astype(int)
    j = (v * (texture_data.shape[0] - 1)).astype(int)

    # Extraction vectorisée des pixels
    colors = texture_data[j, i, :3] / 255.0
    # On suppose que NumpyRGBColor peut être construit en passant les trois tableaux correspondant aux composantes R, G et B.
    return NumpyRGBColor(colors[..., 0], colors[..., 1], colors[..., 2])
