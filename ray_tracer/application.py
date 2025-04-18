from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from ray_tracer.domain.models import Camera, Scene3D, Shape, Vector3D


class Renderer(ABC):
    """The renderer calculates the direction of the rays to be traced, traces the rays
    and saves the result.
    """

    @abstractmethod
    def raytrace_scene(
        self,
        ray_origin: Vector3D,
        normalized_ray_direction: list[Vector3D],
        scene: Scene3D,
    ) -> Vector3D:
        pass

    @abstractmethod
    def get_ray_directions(self, camera: Camera) -> list[Vector3D]:
        pass

    @abstractmethod
    def save_image(
        self,
        color: Vector3D,
        camera: Camera,
        output_path: Path,
    ) -> None:
        pass


class Shader(ABC):
    @abstractmethod
    def to_rgb(self, shader: Self) -> Vector3D:
        pass

    @abstractmethod
    def create(
        self,
        shape: Shape,
        scene: Scene3D,
        ray_origin: Vector3D,
        normalized_ray_direction: Vector3D,
        distance_origin_to_intersection: float,
        ray_tracer: Renderer,
    ) -> Vector3D:
        """This is the main function that will be called to compute the color of the pixel."""
        pass

    @abstractmethod
    def _calculate_shadow(
        self,
        nudged_intersection_point: Vector3D,
        direction_to_light: Vector3D,
        scene: Scene3D,
        shape: Shape,
    ) -> bool:
        pass

    @abstractmethod
    def _calculate_diffuse(
        self,
        intersection_point: Vector3D,
        normal: Vector3D,
        direction_to_light: Vector3D,
        is_in_light: bool,
        shape: Shape,
    ) -> Vector3D:
        """Calculates the diffuse color of the shape at the intersection point."""
        pass

    @abstractmethod
    def _calculate_reflection(  # TODO: VOIR TRANSMISSION
        self,
        nudged_intersection_point: Vector3D,
        normal: Vector3D,
        normalized_ray_direction: Vector3D,
        scene: Scene3D,
        ray_tracer: Renderer,
    ) -> Vector3D:
        pass

    @abstractmethod
    def _calculate_specular(
        self,
        normal: Vector3D,
        direction_to_light: Vector3D,
        direction_to_ray_origin: Vector3D,
    ) -> Vector3D:
        """Calculates the Phong specular reflection based on the angle between the normal and the direction
        to the light source.
        """
        pass

    @abstractmethod
    def _calculate_iridescence(  # TODO: NE SE VOIT QUE DANS LE SPECULAIRE!
        self,
        normal: Vector3D,
        direction_to_ray_origin: Vector3D,
    ) -> Vector3D:
        """Calculates an iridescence color based on thin-film interference,
        incorporating three parameters:
            - thin_film_weight: the weight (coverage) that blends the film effect with the base,
            - thin_film_thickness: the thickness of the film, affecting the spacing of the fringes,
            - thin_film_ior: the refractive index of the film, influencing the hue of the fringes.

        The effect is achieved by modulating a sinusoidal oscillation (simulating interference)
        based on the angle between the normal and the direction to the ray origin.
        """
        pass

    @abstractmethod
    def get_texture_color(self, texture, u: list, v: list) -> Vector3D:
        pass


def render_image_pipeline(
    scene: Scene3D,
    output_path: Path,
    render_service: Renderer,
) -> None:
    normalized_ray_destinations = render_service.get_ray_directions(scene.camera)

    color = render_service.raytrace_scene(scene.camera.position, normalized_ray_destinations, scene)

    render_service.save_image(color, scene.camera, output_path)
