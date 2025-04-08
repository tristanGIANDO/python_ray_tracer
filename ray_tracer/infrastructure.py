import numbers
from functools import reduce
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image

from ray_tracer.application import Renderer, Shader
from ray_tracer.domain import Camera, DomeLight, RGBColor, Scene3D, Shape, Vector3D

FARAWAY = 1.0e39


def extract(cond: ArrayLike, x: numbers.Number | np.ndarray) -> numbers.Number | np.ndarray:
    """Extracts elements from an array or returns the number itself if it is a scalar.

    Args:
        cond (array-like): Condition array to extract elements.
        x: Input number or array.
    """
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


class NumpyVector3D(Vector3D):
    """Represents a 3D vector with numpy-based operations."""

    def __init__(self, x: int | float, y: int | float, z: int | float) -> None:
        (self.x, self.y, self.z) = (x, y, z)

    def dot(self, other: Self) -> float:
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self) -> float:
        return self.dot(self)

    def components(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __mul__(self, other: int | float) -> Self:
        return NumpyVector3D(self.x * other, self.y * other, self.z * other)

    def __add__(self, other: Self) -> Self:
        return NumpyVector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Self) -> Self:
        return NumpyVector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def norm(self) -> Any:
        """Normalizes the vector to have a magnitude of 1."""
        mag = np.sqrt(self.dot(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def extract(self, cond: ArrayLike) -> Self | "NumpyVectorArray3D":
        """Extracts components of the vector based on a condition."""
        if isinstance(cond, numbers.Number):
            return self

        return NumpyVectorArray3D(extract(cond, self.x), extract(cond, self.y), extract(cond, self.z))

    def place(self, cond: ArrayLike) -> Self:
        """Places the vector's components into a new vector."""
        r = NumpyVector3D(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r


class NumpyVectorArray3D(NumpyVector3D):
    pass


class NumpyRGBColor(NumpyVector3D):
    pass


class Texture:
    def __init__(self, color: NumpyRGBColor | None) -> None:
        self.color = color if color is not None else NumpyRGBColor(1, 1, 1)

    def get_color(self, intersection_point: NumpyVectorArray3D) -> NumpyRGBColor:
        """Returns the color of the texture at the intersection point."""
        return self.color


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
        normal = (intersection_point - self.shape.position) * (1.0 / self.shape.radius)
        direction_to_light = (self.scene.lights[0].position - intersection_point).norm()
        direction_to_ray_origin = (self.scene.camera.position - intersection_point).norm()
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

    def diffusecolor(self, intersection_point: NumpyVectorArray3D) -> NumpyVector3D:
        return self.diffuse


class NumpyTexturedSphere(NumpySphere):
    def __init__(
        self,
        center: NumpyVector3D,
        radius: float,
        texture_path: Path,
    ) -> None:
        super().__init__(center, radius, NumpyRGBColor(1, 1, 1))
        image = Image.open(texture_path).convert("RGB")
        self.texture = np.asarray(image) / 255.0

    def diffusecolor(self, intersection_point: NumpyVectorArray3D) -> NumpyRGBColor:
        normal = (intersection_point - self.center).norm()

        u = 0.5 + np.arctan2(normal.z, normal.x) / (2 * np.pi)
        v = 0.5 - np.arcsin(normal.y) / np.pi

        u = u % 1  # Repeat if u > 1
        v = v % 1
        height, width, _ = self.texture.shape

        i = (u * (width - 1)).astype(int) if isinstance(u, np.ndarray) else int(u * (width - 1))
        j = (v * (height - 1)).astype(int) if isinstance(v, np.ndarray) else int(v * (height - 1))

        color = self.texture[j, i, :3]
        if len(color.shape) > 1 and color.shape[0] > 1:
            color = color.mean(axis=0)  # Reduce to a single RGB value by averaging
        if color.shape != (3,):
            if color.shape == (1, 3):
                color = color.flatten()
            else:
                raise ValueError(f"Expected 3 color channels, but got {color.shape}")
        r, g, b = color
        return NumpyRGBColor(r, g, b)


class TextureChecker(Texture):
    def __init__(
        self,
        color: NumpyRGBColor | None = None,
    ) -> None:
        pass

    def get_color(self, intersection_point: NumpyVectorArray3D) -> NumpyRGBColor:
        checker = ((intersection_point.x * 2).astype(int) % 2) == ((intersection_point.z * 2).astype(int) % 2)

        return NumpyRGBColor(1, 1, 1) * checker


class NumpyRenderer(Renderer):
    def raytrace_scene(
        self,
        ray_origin: NumpyVector3D,
        normalized_ray_direction: NumpyVector3D,
        scene: Scene3D,
    ) -> NumpyRGBColor:
        distances = [shape.intersect(ray_origin, normalized_ray_direction) for shape in scene.shapes]
        nearest_distance = reduce(np.minimum, distances)

        color = NumpyRGBColor(0, 0, 0)

        for shape, distance in zip(scene.shapes, distances, strict=False):
            hit: bool = (nearest_distance != FARAWAY) & (distance == nearest_distance)

            if np.any(hit):  # keep only the intersected points
                extracted_distance_to_intersection: np.ndarray = extract(hit, distance)
                extracted_ray_origin: NumpyVector3D = ray_origin.extract(hit)
                extracted_normalized_ray_direction: NumpyVectorArray3D = normalized_ray_direction.extract(hit)

                computed_color = shape.shader.create(
                    shape,
                    scene,
                    extracted_ray_origin,
                    extracted_normalized_ray_direction,
                    extracted_distance_to_intersection,
                    self,
                )

                color += computed_color.place(hit)

        return color

    def get_ray_directions(self, camera: Camera) -> np.ndarray:
        """1. Adjusts the aspect ratio of the camera to fit the image desired size.
        2. Creates a grid of points in the camera's view frustum (like pixels on a screen).
        3. Normalizes the grid points to create ray directions.
        4. Returns the normalized ray directions.
        The grid is created by tiling and repeating the x and y coordinates based on the camera's width and height.
        """
        aspect_ratio = float(camera.width) / camera.height
        screen = (-1, 1 / aspect_ratio + 0.25, 1, -1 / aspect_ratio + 0.25)
        x = np.tile(
            np.linspace(screen[0], screen[2], camera.width),
            camera.height,
        )
        y = np.repeat(
            np.linspace(screen[1], screen[3], camera.height),
            camera.width,
        )

        return (NumpyVector3D(x, y, 0) - camera.position).norm()

    def save_image(self, color: RGBColor, camera: Camera, output_path: str) -> None:
        rgb_colors = [
            Image.fromarray(
                (255 * np.clip(layer, 0, 1).reshape((camera.height, camera.width))).astype(np.uint8),
                "L",
            )
            for layer in color.components()
        ]
        Image.merge("RGB", rgb_colors).save(output_path)
