from functools import reduce

import numpy as np

from ray_tracer.domain import (
    Camera,
    PointLight,
    RGBColor,
    Scene3D,
    Shape,
    Vector3D,
    extract,
)
from ray_tracer.services import RayTracer, Shader

FARAWAY = 1.0e39


class NumpyShader(Shader):
    def __init__(
        self,
        shape: Shape,
        light: PointLight,
        camera: Camera,
        ray_tracer: RayTracer,
        ray_origin: Vector3D,
        normalized_ray_direction: Vector3D,
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

    def create(self) -> RGBColor:
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
        nudged_intersection_point: Vector3D,
        direction_to_light: Vector3D,
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
        intersection_point: Vector3D,
        normal: Vector3D,
        direction_to_light: Vector3D,
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
        ray_tracer: RayTracer,
        nudged_intersection_point: Vector3D,
        normal: Vector3D,
        normalized_ray_direction: Vector3D,
        scene: Scene3D,
        mirror: float,
        reflection_gain: float,
        specular_gain: float,
        camera: Camera,
        light: PointLight,
    ) -> RGBColor:
        rayD = (
            normalized_ray_direction - normal * 2 * normalized_ray_direction.dot(normal)
        ).norm()
        return (
            ray_tracer.render(
                camera,
                light,
                nudged_intersection_point,
                rayD,
                scene,
                reflection_gain,
                specular_gain,
            )
            * mirror
        )

    def calculate_phong_specular(
        self,
        normal: Vector3D,
        direction_to_light: Vector3D,
        direction_to_ray_origin: Vector3D,
    ):
        phong = (direction_to_light + direction_to_ray_origin).norm()
        return RGBColor(1, 1, 1) * np.power(np.clip(normal.dot(phong), 0, 1), 50)

    def calculate_ambient_color(self) -> RGBColor:
        return RGBColor(0.05, 0.05, 0.05)


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
        ray_tracer: RayTracer,
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


class NumpyRayTracer(RayTracer):
    def render(
        self,
        camera: Camera,
        light: PointLight,
        ray_origin: Vector3D,
        normalized_ray_direction: Vector3D,
        scene: Scene3D,
        reflection_gain: float,
        specular_gain: float,
    ) -> RGBColor:
        distances = [
            shape.intersect(ray_origin, normalized_ray_direction)
            for shape in scene.shapes
        ]
        nearest_distance = reduce(np.minimum, distances)

        color = RGBColor(0, 0, 0)

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
