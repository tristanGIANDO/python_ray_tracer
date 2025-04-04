from abc import ABC, abstractmethod
from pathlib import Path

from ray_tracer.domain import Camera, RGBColor, Scene3D, Shape, Vector3D


class RenderService(ABC):
    @abstractmethod
    def render_scene(
        self,
        ray_origin: Vector3D,
        normalized_ray_direction: Vector3D,
        scene: Scene3D,
        reflection_gain: float,
        specular_gain: float,
    ) -> RGBColor:
        pass

    @abstractmethod
    def get_rays_destinations(self, camera: Camera) -> list[Vector3D]:
        pass

    @abstractmethod
    def save_image(
        self,
        color: RGBColor,
        camera: Camera,
        output_path: Path,
    ) -> None:
        pass


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


def render_image_pipeline(
    scene: Scene3D,
    output_path: Path,
    render_service: RenderService,
    reflection_gain: float,
    specular_gain: float,
) -> None:
    normalized_ray_destinations = render_service.get_rays_destinations(scene.camera)

    color = render_service.render_scene(
        scene.camera.position,
        normalized_ray_destinations,
        scene,
        reflection_gain,
        specular_gain,
    )

    render_service.save_image(color, scene.camera, output_path)
