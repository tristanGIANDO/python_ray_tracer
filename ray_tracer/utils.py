import numpy as np
from PIL import Image

from ray_tracer.vectors import Vector3D


def get_texture_color(texture: Image, u: float, v: float) -> Vector3D:
    """
    Retrieves the color from a texture image at given UV coordinates.

    Args:
        texture (Image): The texture image.
        u (float): U coordinate (horizontal).
        v (float): V coordinate (vertical).

    Returns:
        Vector3D: The color at the specified UV coordinates, normalized between 0 and 1.

    The UV coordinates (u, v) are like X and Y positions on the texture image. This function maps those coordinates to the
    actual pixel in the image, allowing the color to be sampled. The colors are then normalized to be between 0 and 1.
    """
    texture_data = np.array(texture)
    u = u % 1  # Repeat if u exceeds 1
    v = v % 1
    i = int(u * (texture_data.shape[1] - 1))
    j = int(v * (texture_data.shape[0] - 1))
    return Vector3D(*texture_data[j, i, :3] / 255)


class HDRIEnvironment:
    def __init__(self, hdr_image_path: str):
        self.image = Image.open(hdr_image_path)
        self.image_data = np.array(self.image) / 255.0  # Normaliser entre 0 et 1
        self.width = self.image_data.shape[1]
        self.height = self.image_data.shape[0]

    def get_color(self, direction: Vector3D) -> Vector3D:
        """
        Récupère la couleur de l'environnement en fonction de la direction du rayon.

        Args:
            direction (Vector3D): La direction du rayon.

        Returns:
            Vector3D: La couleur de l'environnement.
        """
        # Convertir le vecteur en coordonnées sphériques
        theta = np.arccos(direction.y)  # Angle de l'axe Y
        phi = np.arctan2(direction.z, direction.x)  # Angle autour de l'axe Y

        # Normaliser phi pour qu'il soit entre [0, 1]
        u = (phi + np.pi) / (2 * np.pi)
        v = theta / np.pi

        # Convertir (u, v) en coordonnées d'image
        i = int(u * (self.width - 1))
        j = int((1 - v) * (self.height - 1))

        # Récupérer la couleur et retourner comme vecteur
        color = self.image_data[j, i, :3]
        return Vector3D(*color)
