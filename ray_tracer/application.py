from abc import ABC, abstractmethod
from pathlib import Path

from ray_tracer.domain import Camera, RGBColor, Scene3D, Vector3D


class Renderer(ABC):
    """The renderer calculates the direction of the rays to be traced, traces the rays
    and saves the result.
    """

    @abstractmethod
    def raytrace_scene(
        self,
        ray_origin: Vector3D,
        normalized_ray_direction: Vector3D,
        scene: Scene3D,
    ) -> RGBColor:
        pass

    @abstractmethod
    def get_ray_directions(self, camera: Camera) -> list[Vector3D]:
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
    """Handles shading calculations for a ray-traced scene."""

    @abstractmethod
    def create(self) -> Vector3D:
        pass


def render_image_pipeline(
    scene: Scene3D,
    output_path: Path,
    render_service: Renderer,
) -> None:
    normalized_ray_destinations = render_service.get_ray_directions(scene.camera)

    color = render_service.raytrace_scene(scene.camera.position, normalized_ray_destinations, scene)

    render_service.save_image(color, scene.camera, output_path)
