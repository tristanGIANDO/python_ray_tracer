from functools import reduce
from typing import Self

import numpy as np

from ray_tracer.application import Renderer, Shader
from ray_tracer.domain import DomeLight, Scene3D, Shape
from ray_tracer.infrastructure.numpy.shape import NumpyRGBColor, NumpyVector3D, NumpyVectorArray3D

FARAWAY = 1.0e39


class Texture:
    def __init__(self, color: NumpyRGBColor | None = None) -> None:
        self.color = color if color is not None else NumpyRGBColor(1, 1, 1)

    def get_color(self, intersection_point: NumpyVectorArray3D) -> NumpyRGBColor:
        """Returns the color of the texture at the intersection point."""
        return self.color


class TextureChecker(Texture):
    def __init__(
        self,
        color: NumpyRGBColor | None = None,
    ) -> None:
        self.color = color if color is not None else NumpyRGBColor(1, 1, 1)

    def get_color(self, intersection_point: NumpyVectorArray3D) -> NumpyRGBColor:
        checker = ((intersection_point.x * 2).astype(int) % 2) == ((intersection_point.z * 2).astype(int) % 2)

        return NumpyRGBColor(1, 1, 1) * checker


class NumpyShader(Shader):
    def __init__(
        self,
        reflection_gain: float,
        specular_gain: float,
        specular_roughness: float,
        iridescence_gain: float,
        diffuse_gain: float,
        diffuse_color: Texture,
    ):
        self.reflection_gain = reflection_gain
        self.specular_gain = specular_gain
        self.specular_roughness = specular_roughness
        self.iridescence_gain = iridescence_gain
        self.diffuse_gain = diffuse_gain
        self.diffuse_color = diffuse_color

    def to_rgb(self, shader: Self) -> NumpyRGBColor:
        return NumpyRGBColor(
            np.clip(shader.x, 0, 1),
            np.clip(shader.y, 0, 1),
            np.clip(shader.z, 0, 1),
        )

    def create(
        self,
        shape: Shape,
        scene: Scene3D,
        ray_origin: NumpyVector3D,
        normalized_ray_direction: NumpyVector3D,
        distance_origin_to_intersection: float,
        ray_tracer: Renderer,
    ) -> NumpyVector3D:
        """This is the main function that will be called to compute the color of the pixel."""
        intersection_point = ray_origin + normalized_ray_direction * distance_origin_to_intersection
        normal = (intersection_point - shape.position) * (1.0 / shape.radius)
        direction_to_light = (scene.lights[0].position - intersection_point).norm()
        direction_to_ray_origin = (scene.camera.position - intersection_point).norm()
        nudged_intersection_point = intersection_point + normal * 0.0001  # to avoid itself

        is_in_light = self._calculate_shadow(
            nudged_intersection_point,
            direction_to_light,
            scene,
            shape,
        )

        color = self._calculate_ambient_color()

        color += self._calculate_diffuse(intersection_point, normal, direction_to_light, is_in_light)

        color += self._calculate_reflection(
            nudged_intersection_point,
            normal,
            normalized_ray_direction,
            scene,
            ray_tracer,
        )

        color += self._calculate_dome_light(normal, scene)

        color += self._calculate_phong_specular(normal, direction_to_light, direction_to_ray_origin) * is_in_light

        color += self._calculate_iridescence(normal, direction_to_ray_origin)

        return color

    def _calculate_shadow(
        self,
        nudged_intersection_point: NumpyVector3D,
        direction_to_light: NumpyVector3D,
        scene: Scene3D,
        shape: Shape,
    ) -> bool:
        """Calculates whether the intersection point is in light or in shadow.
        It does this by checking if the intersection point is closer to the light source.
        If the intersection point is closer to the light source than any other shape, it is in light.
        Otherwise, it is in shadow.
        """
        light_distances = [shape.intersect(nudged_intersection_point, direction_to_light) for shape in scene.shapes]
        light_nearest = reduce(np.minimum, light_distances)
        return light_distances[scene.shapes.index(shape)] == light_nearest

    def _calculate_diffuse(
        self,
        intersection_point: NumpyVectorArray3D,
        normal: NumpyVector3D,
        direction_to_light: NumpyVector3D,
        is_in_light: bool,
    ) -> NumpyRGBColor:
        """Calculates the diffuse color of the shape at the intersection point."""
        diffuse_light_intensity = np.maximum(normal.dot(direction_to_light), 0)
        return (
            self.diffuse_color.get_color(intersection_point) * diffuse_light_intensity * is_in_light * self.diffuse_gain
        )

    def _calculate_reflection(
        self,
        nudged_intersection_point: NumpyVector3D,
        normal: NumpyVector3D,
        normalized_ray_direction: NumpyVector3D,
        scene: Scene3D,
        ray_tracer: Renderer,
    ) -> NumpyRGBColor:
        ray_direction = (normalized_ray_direction - normal * 2 * normalized_ray_direction.dot(normal)).norm()
        return (
            ray_tracer.raytrace_scene(
                nudged_intersection_point,
                ray_direction,
                scene,
            )
            * self.reflection_gain
        )

    def _calculate_phong_specular(
        self,
        normal: NumpyVector3D,
        direction_to_light: NumpyVector3D,
        direction_to_ray_origin: NumpyVector3D,
    ) -> NumpyRGBColor:
        """Calculates the Phong specular reflection based on the angle between the normal and the direction
        to the light source.
        """
        phong = (direction_to_light + direction_to_ray_origin).norm()
        unclamped_roughness = (
            (1 - self.specular_roughness) * 100
        )  # more user friendly to have roughness between 0 and 1 where 0 means no roughness and 1 means very rough

        return (
            NumpyRGBColor(1, 1, 1)
            * np.power(np.clip(normal.dot(phong), 0, 1), unclamped_roughness)
            * self.specular_gain
        )

    def _calculate_ambient_color(self) -> NumpyRGBColor:
        return NumpyRGBColor(0.004, 0.004, 0.004)  # minimum black color

    def _calculate_iridescence(
        self,
        normal: NumpyVector3D,
        direction_to_ray_origin: NumpyVector3D,
    ) -> NumpyRGBColor:
        """Irisdescence effect based on the angle between the normal and the direction to the ray origin.
        The more the angle is close to 90 degrees, the more the iridescence effect is visible.
        The iridescence effect is more pronounced at the edges of the object.
        """
        view_angle = np.clip(normal.dot(direction_to_ray_origin), 0.0, 1.0)

        iridescence_factor = (
            np.abs(view_angle - 0.5) * 2  # 0.5 is the middle of the angle, it amplifies the effect on the edges
        )

        color_variation = np.sin(iridescence_factor * np.pi)
        iridescence_color = NumpyRGBColor(color_variation, 1 - color_variation, 0.5)  # modify z to shift the color

        return iridescence_color * self.iridescence_gain

    def _calculate_dome_light(self, normal: NumpyVector3D, scene: Scene3D) -> NumpyRGBColor:
        light_direction = NumpyVector3D(0, 1, 0)  # from sky
        dome_intensity = 0.0
        dome_color = NumpyRGBColor(1.0, 1.0, 1.0)

        for light in scene.lights:
            if isinstance(light, DomeLight):
                dome_color = light.color
                dome_intensity += light.intensity * np.maximum(normal.dot(light_direction), 0)

        return dome_color * dome_intensity

    def _calculate_physical_specular(
        self,
        normal: np.ndarray,
        direction_to_light: np.ndarray,
        direction_to_ray_origin: np.ndarray,
        base_weight: float = 1.0,
        base_color: np.ndarray = np.array([1.0, 1.0, 1.0]),
        specular_weight: float = 1.0,
        specular_color: np.ndarray = np.array([1.0, 1.0, 1.0]),
        specular_roughness: float = 0.01,
        specular_roughness_anisotropy: float = 0.0,
    ) -> np.ndarray:
        """Calcule le BRDF spéculaire d’un métal avec Fresnel F82-tint et modèle GGX."""

        def normalize(v):
            return v / (np.linalg.norm(v) + 1e-8)

        def dot(a, b):
            return np.clip(np.dot(a.flatten(), b.flatten()), 0.0, 1.0)

        def F_schlick(mu, F0):
            return F0 + (1 - F0) * (1 - mu) ** 5

        def compute_fresnel_F82(mu, F0, F_schlick_bar, F_bar, specular_weight):
            correction = mu * (1 - mu) ** 6 * mu_bar * (1 - mu_bar) ** 6 * (F_schlick_bar - F_bar)
            return specular_weight * (F_schlick(mu, F0) - correction)

        def D_GGX(NdotH, alpha):
            denom = NdotH**2 * (alpha**2 - 1) + 1
            return (alpha**2) / (np.pi * denom**2 + 1e-8)

        def G_smith(NdotV, NdotL, alpha):
            def G1(NdotX):
                a = alpha
                return 2 * NdotX / (NdotX + np.sqrt(a**2 + (1 - a**2) * NdotX**2 + 1e-8))

            return G1(NdotV) * G1(NdotL)

        normal = np.array(normal.components())
        direction_to_light = np.array(direction_to_light.components())
        direction_to_ray_origin = np.array(direction_to_ray_origin.components())

        # Vecteurs normalisés
        n = normalize(normal)
        l = normalize(direction_to_light)
        v = normalize(direction_to_ray_origin)

        # μ = dot(normal, view direction)
        mu = np.clip(dot(n, v), 0.0, 1.0)

        # F0 = base_weight * base_color
        F0 = base_weight * base_color

        # F_schlick(μ)
        F_schlick_mu = F_schlick(mu, F0)

        # Calcul du terme correctif (F82)
        mu_bar = 1 / 7
        F_schlick_bar = F_schlick(mu_bar, F0)
        F_bar = specular_color * F_schlick_bar

        correction_term = mu * (1 - mu) ** 6 * mu_bar * (1 - mu_bar) ** 6 * (F_schlick_bar - F_bar)

        F82 = F_schlick_mu - correction_term

        # Appliquer le poids global
        F_metal = specular_weight * F82

        x, y, z = np.clip(F_metal, 0.0, 1.0)

        return NumpyRGBColor(x, y, z)
