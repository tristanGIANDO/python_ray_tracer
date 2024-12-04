import numpy as np

from src.vector import Vec3


def get_texture_color(texture, u, v):
    texture_data = np.array(texture)
    # Convertir UV en indices de pixel
    u = u % 1  # Répéter si u dépasse 1
    v = v % 1
    i = int(u * (texture_data.shape[1] - 1))
    j = int(v * (texture_data.shape[0] - 1))
    return Vec3(*texture_data[j, i, :3] / 255)  # Normaliser les couleurs


class Sphere:
    def __init__(self, center, radius, color, reflection=0.5, texture=None):
        self.center = center
        self.radius = radius
        self.color = color
        self.reflection = reflection
        self.texture = texture  # Ajouter une texture optionnelle

    def get_surface_color(self, hit_point):
        if self.texture:
            # Calculer les coordonnées UV
            normal = (hit_point - self.center).norm()
            u = 0.5 + np.arctan2(normal.z, normal.x) / (2 * np.pi)
            v = 0.5 - np.arcsin(normal.y) / np.pi
            return get_texture_color(self.texture, u, v)
        return self.color

    def intersect(self, ray_origin, ray_dir):
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
