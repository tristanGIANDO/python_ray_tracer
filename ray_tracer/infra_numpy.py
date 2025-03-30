import numbers
from functools import reduce

import numpy as np

from ray_tracer.application import RenderService, Shader
from ray_tracer.domain import (
    Camera,
    PointLight,
    RenderImage,
    Scene3D,
    Shape,
    Vector3D,
)

FARAWAY = 1.0e39


def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


class NumpyVector3D(Vector3D):
    def __init__(self, x: int | float, y: int | float, z: int | float) -> None:
        (self.x, self.y, self.z) = (x, y, z)

    def dot(self, other: "NumpyVector3D") -> float:
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self) -> float:
        return self.dot(self)

    def components(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __mul__(self, other: int | float) -> "NumpyVector3D":
        return NumpyVector3D(self.x * other, self.y * other, self.z * other)

    def __add__(self, other: "NumpyVector3D") -> "NumpyVector3D":
        return NumpyVector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "NumpyVector3D") -> "NumpyVector3D":
        return NumpyVector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def norm(self):
        mag = np.sqrt(self.dot(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def extract(self, cond) -> "NumpyVector3D":
        return NumpyVector3D(
            extract(cond, self.x), extract(cond, self.y), extract(cond, self.z)
        )

    def place(self, cond) -> "NumpyVector3D":
        r = NumpyVector3D(
            np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape)
        )
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r


class NumpyRGBColor(NumpyVector3D):
    pass


class NumpyShader(Shader):
    def __init__(
        self,
        shape: Shape,
        light: PointLight,
        camera: Camera,
        ray_tracer: RenderService,
        ray_origin: NumpyVector3D,
        normalized_ray_direction: NumpyVector3D,
        distance_origin_to_intersection,
        scene: Scene3D,
        reflection_gain: float,
        specular_gain: float,
    ):
        self.shape = shape
        self.light = light
        self.camera = camera
        self.ray_tracer = ray_tracer
        self.ray_origin = ray_origin
        self.normalized_ray_direction = normalized_ray_direction
        self.distance_origin_to_intersection = distance_origin_to_intersection
        self.scene = scene
        self.reflection_gain = reflection_gain
        self.specular_gain = specular_gain

    def create(self) -> NumpyRGBColor:
        intersection_point = (
            self.ray_origin
            + self.normalized_ray_direction * self.distance_origin_to_intersection
        )
        normal = (intersection_point - self.shape.position) * (1.0 / self.shape.radius)
        direction_to_light = (self.light.position - intersection_point).norm()
        direction_to_ray_origin = (self.camera.position - intersection_point).norm()
        nudged_intersection_point = (
            intersection_point + normal * 0.0001
        )  # to avoid itself

        is_in_light = self.calculate_shadow(
            self.scene.shapes.index(self.shape),
            self.scene.shapes,
            nudged_intersection_point,
            direction_to_light,
        )

        color = self.calculate_ambient_color()

        color += self.calculate_diffuse(
            self.shape, intersection_point, normal, direction_to_light, is_in_light
        )

        color += (
            self.calculate_reflection(
                self.ray_tracer,
                nudged_intersection_point,
                normal,
                self.normalized_ray_direction,
                self.scene,
                self.shape.mirror,
                self.reflection_gain,
                self.specular_gain,
                self.camera,
                self.light,
            )
            * self.reflection_gain
        )

        color += (
            self.calculate_phong_specular(
                normal, direction_to_light, direction_to_ray_origin
            )
            * self.specular_gain
            * is_in_light
        )

        return color

    def calculate_shadow(
        self,
        shape_index: int,
        shapes: list[Shape],
        nudged_intersection_point: NumpyVector3D,
        direction_to_light: NumpyVector3D,
    ) -> bool:
        light_distances = [
            shape.intersect(nudged_intersection_point, direction_to_light)
            for shape in shapes
        ]
        light_nearest = reduce(np.minimum, light_distances)
        return light_distances[shape_index] == light_nearest

    def calculate_diffuse(
        self,
        shape: Shape,
        intersection_point: NumpyVector3D,
        normal: NumpyVector3D,
        direction_to_light: NumpyVector3D,
        is_in_light: bool,
    ):
        diffuse_light_intensity = np.maximum(normal.dot(direction_to_light), 0)
        return (
            shape.diffusecolor(intersection_point)
            * diffuse_light_intensity
            * is_in_light
        )

    def calculate_reflection(
        self,
        ray_tracer: RenderService,
        nudged_intersection_point: NumpyVector3D,
        normal: NumpyVector3D,
        normalized_ray_direction: NumpyVector3D,
        scene: Scene3D,
        mirror: float,
        reflection_gain: float,
        specular_gain: float,
        camera: Camera,
        light: PointLight,
    ) -> NumpyRGBColor:
        ray_direction = (
            normalized_ray_direction - normal * 2 * normalized_ray_direction.dot(normal)
        ).norm()
        return (
            ray_tracer.render_scene(
                camera,
                light,
                nudged_intersection_point,
                ray_direction,
                scene,
                reflection_gain,
                specular_gain,
            )
            * mirror
        )

    def calculate_phong_specular(
        self,
        normal: NumpyVector3D,
        direction_to_light: NumpyVector3D,
        direction_to_ray_origin: NumpyVector3D,
    ):
        phong = (direction_to_light + direction_to_ray_origin).norm()
        return NumpyRGBColor(1, 1, 1) * np.power(np.clip(normal.dot(phong), 0, 1), 50)

    def calculate_ambient_color(self) -> NumpyRGBColor:
        return NumpyRGBColor(0.05, 0.05, 0.05)


class NumpySphere(Shape):
    def __init__(
        self,
        center: NumpyVector3D,
        radius: float,
        diffuse: NumpyRGBColor,
        mirror: float = 0.5,
    ) -> None:
        self.position = center
        self.radius = radius
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(
        self, ray_origin: NumpyVector3D, normalized_ray_direction: NumpyVector3D
    ):
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

    def diffusecolor(self, intersection_point) -> NumpyVector3D:
        return self.diffuse

    def create_shader(
        self,
        ray_origin: NumpyVector3D,
        normalized_ray_direction: NumpyVector3D,
        distance_origin_to_intersection,
        scene: Scene3D,
        ray_tracer: RenderService,
        camera: Camera,
        light: PointLight,
        reflection_gain: float,
        specular_gain: float,
    ):
        shader_creator = NumpyShader(
            self,
            light,
            camera,
            ray_tracer,
            ray_origin,
            normalized_ray_direction,
            distance_origin_to_intersection,
            scene,
            reflection_gain,
            specular_gain,
        )
        return shader_creator.create()


class CheckeredSphere(NumpySphere):
    def diffusecolor(self, intersection_point):
        checker = ((intersection_point.x * 2).astype(int) % 2) == (
            (intersection_point.z * 2).astype(int) % 2
        )
        return self.diffuse * checker


class NumpyRenderService(RenderService):
    def render_scene(
        self,
        camera: Camera,
        light: PointLight,
        ray_origin: NumpyVector3D,
        normalized_ray_direction: NumpyVector3D,
        scene: Scene3D,
        reflection_gain: float,
        specular_gain: float,
    ) -> NumpyRGBColor:
        distances = [
            shape.intersect(ray_origin, normalized_ray_direction)
            for shape in scene.shapes
        ]
        nearest_distance = reduce(np.minimum, distances)

        color = NumpyRGBColor(0, 0, 0)

        for shape, distance in zip(scene.shapes, distances):
            hit: bool = (nearest_distance != FARAWAY) & (distance == nearest_distance)

            if np.any(hit):  # keep only the intersected points
                extracted_distance_to_intersection = extract(hit, distance)
                extracted_ray_origin = ray_origin.extract(hit)
                extracted_normalized_ray_direction = normalized_ray_direction.extract(
                    hit
                )

                computed_color = shape.create_shader(
                    extracted_ray_origin,
                    extracted_normalized_ray_direction,
                    extracted_distance_to_intersection,
                    scene,
                    self,
                    camera,
                    light,
                    reflection_gain,
                    specular_gain,
                )

                color += computed_color.place(hit)

        return color


def get_rays_destinations(render_image: RenderImage, camera: Camera) -> np.ndarray:
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

    ray_destinations = NumpyVector3D(x, y, 0)

    return (ray_destinations - camera.position).norm()
