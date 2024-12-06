import numpy as np

from src.vectors import Vector3D


def get_texture_color(texture, u: float, v: float) -> Vector3D:
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
