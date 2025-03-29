import numbers
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
from PIL import Image


def extract(cond, x) -> numbers.Number | np.ndarray:
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


class Vector3D:
    def __init__(self, x: int | float, y: int | float, z: int | float) -> None:
        (self.x, self.y, self.z) = (x, y, z)

    def __mul__(self, other) -> "Vector3D":
        return Vector3D(self.x * other, self.y * other, self.z * other)

    def __add__(self, other) -> "Vector3D":
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other) -> "Vector3D":
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self) -> int | float:
        return self.dot(self)

    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def components(self) -> tuple[int | float, int | float, int | float]:
        return (self.x, self.y, self.z)

    def extract(self, cond) -> "Vector3D":
        return Vector3D(
            extract(cond, self.x), extract(cond, self.y), extract(cond, self.z)
        )

    def place(self, cond) -> "Vector3D":
        r = Vector3D(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r


RGBColor = Vector3D

FARAWAY = 1.0e39  # an implausibly huge distance


@dataclass
class Camera:
    position: Vector3D


@dataclass
class PointLight:
    position: Vector3D


@dataclass
class RenderImage:
    width: int
    height: int


class Shape(ABC):
    @abstractmethod
    def intersect(self, ray_origin: Vector3D, normalized_ray_direction: Vector3D):
        pass

    @abstractmethod
    def diffusecolor(self, intersection_point: Vector3D):
        pass

    @abstractmethod
    def create_shader(
        self,
        ray_origin: Vector3D,
        normalized_ray_direction: Vector3D,
        distance_origin_to_intersection,
        scene,
        reflection_gain: float,
        specular_gain: float,
    ):
        pass


@dataclass
class Scene3D:
    shapes: list[Shape]


def raytrace(
    ray_origin: Vector3D,
    normalized_ray_direction: Vector3D,
    scene: Scene3D,
    reflection_gain: float,
    specular_gain: float,
) -> RGBColor:
    distances = [
        shape.intersect(ray_origin, normalized_ray_direction) for shape in scene.shapes
    ]
    nearest_distance = reduce(np.minimum, distances)

    color = RGBColor(0, 0, 0)

    for shape, distance in zip(scene.shapes, distances):
        hit: bool = (nearest_distance != FARAWAY) & (distance == nearest_distance)

        if np.any(hit):  # keep only the intersected points
            extracted_distance_to_intersection = extract(hit, distance)
            extracted_ray_origin = ray_origin.extract(hit)
            extracted_normalized_ray_direction = normalized_ray_direction.extract(hit)

            computed_color = shape.create_shader(
                extracted_ray_origin,
                extracted_normalized_ray_direction,
                extracted_distance_to_intersection,
                scene,
                reflection_gain,
                specular_gain,
            )

            color += computed_color.place(hit)

    return color


class NumpySphere(Shape):
    def __init__(
        self, center: Vector3D, radius: float, diffuse: RGBColor, mirror: float = 0.5
    ) -> None:
        self.position = center
        self.radius = radius
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(self, ray_origin: Vector3D, normalized_ray_direction: Vector3D):
        """Cette fonction calcule l'intersection entre un rayon (défini par son origine
        et sa direction) et une sphère (définie par self.position et self.radius).
        Elle retourne la distance entre l'origine du rayon et le point d'intersection,
        ou une valeur spéciale (FARAWAY) si aucune intersection n'existe.
        """
        projected_ray_direction = 2 * normalized_ray_direction.dot(
            ray_origin - self.position
        )
        ajusted_squared_distance = (
            abs(self.position)
            + abs(ray_origin)
            - 2 * self.position.dot(ray_origin)
            - (self.radius * self.radius)
        )
        discriminator = (projected_ray_direction**2) - (
            4 * ajusted_squared_distance
        )  # if < 0, no intersection
        discriminator_square_root = np.sqrt(np.maximum(0, discriminator))

        potential_solution_0 = (
            -projected_ray_direction - discriminator_square_root
        ) / 2
        potential_solution_1 = (
            -projected_ray_direction + discriminator_square_root
        ) / 2

        solution = np.where(
            (potential_solution_0 > 0) & (potential_solution_0 < potential_solution_1),
            potential_solution_0,
            potential_solution_1,
        )

        condition_to_have_intersection = (discriminator > 0) & (solution > 0)
        return np.where(condition_to_have_intersection, solution, FARAWAY)

    def diffusecolor(self, intersection_point) -> Vector3D:
        return self.diffuse

    def create_shader(
        self,
        ray_origin: Vector3D,
        normalized_ray_direction: Vector3D,
        distance_origin_to_intersection,
        scene: Scene3D,
        reflection_gain: float,
        specular_gain: float,
    ):
        intersection_point = (
            ray_origin + normalized_ray_direction * distance_origin_to_intersection
        )
        normal = (intersection_point - self.position) * (1.0 / self.radius)
        direction_to_light = (light.position - intersection_point).norm()
        direction_to_ray_origin = (camera.position - intersection_point).norm()
        nudged_intersection_point = (
            intersection_point + normal * 0.0001
        )  # to avoid itself

        is_in_light = calculate_shadow(
            scene.shapes.index(self), nudged_intersection_point, direction_to_light
        )

        color = calculate_ambient_color()

        color += calculate_diffuse(
            self, intersection_point, normal, direction_to_light, is_in_light
        )

        color += (
            calculate_reflection(
                nudged_intersection_point,
                normal,
                normalized_ray_direction,
                scene,
                self.mirror,
                reflection_gain,
                specular_gain,
            )
            * reflection_gain
        )

        color += (
            calculate_phong_specular(
                normal, direction_to_light, direction_to_ray_origin
            )
            * specular_gain
            * is_in_light
        )

        return color


def calculate_shadow(
    shape_index: int, nudged_intersection_point: Vector3D, direction_to_light: Vector3D
) -> bool:
    light_distances = [
        shape.intersect(nudged_intersection_point, direction_to_light)
        for shape in scene.shapes
    ]
    light_nearest = reduce(np.minimum, light_distances)
    return light_distances[shape_index] == light_nearest


def calculate_diffuse(
    shape: Shape,
    intersection_point: Vector3D,
    normal: Vector3D,
    direction_to_light: Vector3D,
    is_in_light: bool,
):
    diffuse_light_intensity = np.maximum(normal.dot(direction_to_light), 0)
    return (
        shape.diffusecolor(intersection_point) * diffuse_light_intensity * is_in_light
    )


def calculate_reflection(
    nudged_intersection_point: Vector3D,
    normal: Vector3D,
    normalized_ray_direction: Vector3D,
    scene: Scene3D,
    mirror: float,
    reflection_gain: float,
    specular_gain: float,
) -> RGBColor:
    rayD = (
        normalized_ray_direction - normal * 2 * normalized_ray_direction.dot(normal)
    ).norm()
    return (
        raytrace(nudged_intersection_point, rayD, scene, reflection_gain, specular_gain)
        * mirror
    )


def calculate_phong_specular(
    normal: Vector3D, direction_to_light: Vector3D, direction_to_ray_origin: Vector3D
):
    phong = (direction_to_light + direction_to_ray_origin).norm()
    return RGBColor(1, 1, 1) * np.power(np.clip(normal.dot(phong), 0, 1), 50)


def calculate_ambient_color() -> RGBColor:
    return RGBColor(0.05, 0.05, 0.05)


class CheckeredSphere(NumpySphere):
    def diffusecolor(self, intersection_point):
        checker = ((intersection_point.x * 2).astype(int) % 2) == (
            (intersection_point.z * 2).astype(int) % 2
        )
        return self.diffuse * checker


def main(
    render_image: RenderImage,
    camera: Camera,
    scene: Scene3D,
    output_path: Path,
    reflection_gain: float,
    specular_gain: float,
) -> None:
    aspect_ratio = float(render_image.width) / render_image.height
    projection_canva = (-1, 1 / aspect_ratio + 0.25, 1, -1 / aspect_ratio + 0.25)
    x = np.tile(
        np.linspace(projection_canva[0], projection_canva[2], render_image.width),
        render_image.height,
    )
    y = np.repeat(
        np.linspace(projection_canva[1], projection_canva[3], render_image.height),
        render_image.width,
    )

    start_time = time.time()
    ray_destinations = Vector3D(x, y, 0)
    color = raytrace(
        camera.position,
        (ray_destinations - camera.position).norm(),
        scene,
        reflection_gain,
        specular_gain,
    )
    print("Took", time.time() - start_time)

    rgb_colors = [
        Image.fromarray(
            (
                255
                * np.clip(grey_color, 0, 1).reshape(
                    (render_image.height, render_image.width)
                )
            ).astype(np.uint8),
            "L",
        )
        for grey_color in color.components()
    ]
    Image.merge("RGB", rgb_colors).save(output_path)


if __name__ == "__main__":
    light = PointLight(Vector3D(5, 5, -10))
    camera = Camera(Vector3D(0, 0.35, -1))
    render_image = RenderImage(400, 300)

    scene = Scene3D(
        [
            NumpySphere(Vector3D(0.75, 0.1, 1), 0.6, RGBColor(0, 0, 1)),
            NumpySphere(Vector3D(-0.75, 0.1, 2.25), 0.6, RGBColor(0.5, 0.223, 0.5)),
            NumpySphere(Vector3D(-2.75, 0.1, 3.5), 0.6, RGBColor(1, 0.572, 0.184)),
            CheckeredSphere(
                Vector3D(0, -99999.5, 0), 99999, RGBColor(0.75, 0.75, 0.75), 0.25
            ),
        ]
    )

    output_path = Path("rt3.png")

    main(render_image, camera, scene, output_path, 1.0, 1.0)
