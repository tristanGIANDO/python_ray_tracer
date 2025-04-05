from abc import ABC, abstractmethod

from ray_tracer.domain import Camera, PointLight, RGBColor, Scene3D, Shape, Vector3D


class Shader(ABC):
    @abstractmethod
    def calculate_shadow(
        shape_index: int,
        shapes: list[Shape],
        nudged_intersection_point: Vector3D,
        direction_to_light: Vector3D,
    ) -> bool:
        pass

    @abstractmethod
    def calculate_diffuse(
        shape: Shape,
        intersection_point: Vector3D,
        normal: Vector3D,
        direction_to_light: Vector3D,
        is_in_light: bool,
    ) -> RGBColor:
        pass

    @abstractmethod
    def calculate_reflection(
        nudged_intersection_point: Vector3D,
        normal: Vector3D,
        normalized_ray_direction: Vector3D,
        scene: Scene3D,
        mirror: float,
        reflection_gain: float,
        specular_gain: float,
    ) -> RGBColor:
        pass

    @abstractmethod
    def calculate_phong_specular(
        normal: Vector3D,
        direction_to_light: Vector3D,
        direction_to_ray_origin: Vector3D,
    ) -> RGBColor:
        pass

    @abstractmethod
    def calculate_ambient_color() -> RGBColor:
        pass


class RayTracer(ABC):
    @abstractmethod
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
        pass
