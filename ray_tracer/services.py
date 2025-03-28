from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator

from ray_tracer.domain import RenderConfig, Scene, Vector3D


class ImageService(ABC):
    @abstractmethod
    def from_path(image_path: Path) -> Any:
        pass

    @abstractmethod
    def to_bitmap(self, image: Any, output_path: Path) -> None:
        pass


class RayTracer(ABC):
    @abstractmethod
    def render(self, scene: Scene, render_config: RenderConfig) -> Generator:
        pass

    @abstractmethod
    def to_standard_size(self, image: Any) -> Any:
        pass


class ObjectService(ABC):
    @abstractmethod
    def intersect(self, ray_origin: Vector3D, ray_dir: Vector3D) -> float | None:
        pass

    @abstractmethod
    def get_surface_color(self, hit_point: Vector3D) -> Vector3D:
        pass


class VectorService(ABC):
    @abstractmethod
    def add(self, vector1: Vector3D, vector2: Vector3D) -> Vector3D:
        """
        Adds two vectors component-wise.

        Args:
            other (Vector3D): The vector to add.

        Returns:
            Vector3D: The resulting vector after addition.

        This method takes another vector and adds each component to the corresponding component of this vector.
        """
        pass

    @abstractmethod
    def sub(self, vector1: Vector3D, vector2: Vector3D) -> Vector3D:
        """
        Subtracts one vector from another component-wise.

        Args:
            other (Vector3D): The vector to subtract.

        Returns:
            Vector3D: The resulting vector after subtraction.

        This method subtracts the x, y, and z components of the other vector from this vector.
        """
        pass

    @abstractmethod
    def mul(self, vector: Vector3D, scalar: float | Vector3D) -> Vector3D:
        """
        Multiplies the vector by a scalar or another vector component-wise.

        Args:
            scalar (float or Vector3D): The scalar or vector to multiply.

        Returns:
            Vector3D: The resulting vector after multiplication.

        If `scalar` is a float, each component of the vector is multiplied by this value.
        If `scalar` is another vector, each component is multiplied by the corresponding component of that vector.
        """
        pass

    @abstractmethod
    def truediv(self, vector: Vector3D, scalar: float) -> Vector3D:
        """
        Divise le vecteur par un scalaire.

        Args:
            scalar (float): Le scalaire par lequel diviser.

        Returns:
            Vector3D: Un nouveau vecteur avec chaque composant divisé par le scalaire.
        """
        pass

    @abstractmethod
    def dot(self, vector1: Vector3D, vector2: Vector3D) -> float:
        """
        Computes the dot product of two vectors.

        Args:
            other (Vector3D): The vector to dot with.

        Returns:
            float: The dot product of the two vectors.

        The dot product is calculated as the sum of the products of the corresponding components:
        `x1 * x2 + y1 * y2 + z1 * z2`. This value represents how aligned two vectors are.
        """
        pass

    @abstractmethod
    def norm(self, vector: Vector3D) -> float:
        """
        Normalizes the vector (scales it to have length 1).

        Returns:
            Vector3D: The normalized vector.

        This method calculates the length of the vector and scales all components so that the vector has a length of 1.
        This is useful for directions. If the length is very small (close to zero), a small value is used to prevent division by zero.
        """
        pass

    @abstractmethod
    def components(self, vector: Vector3D) -> tuple[float, float, float]:
        """
        Returns the components of the vector as a tuple.

        Returns:
            tuple: A tuple containing the x, y, and z components of the vector.

        This method is used to easily extract all three components of the vector for further calculations or storage.
        """
        pass

    @abstractmethod
    def perturb(self, vector: Vector3D, roughness: float) -> Vector3D:
        """
        Perturbe ce vecteur en fonction de la roughness en utilisant un échantillonnage pondéré.

        Args:
            roughness (float): Facteur de rugosité (0 = lisse, 1 = rugueux).

        Returns:
            Vector3D: Vecteur perturbé en fonction de la rugosité.
        """
        pass


def render_single_image_pipeline(
    scene: Scene,
    render_config: RenderConfig,
    ray_tracer: RayTracer,
    image_service: ImageService,
) -> bool:
    image = ray_tracer.render(
        scene, render_config
    )  # it's a yield, you could show each step
    image_sized = ray_tracer.to_standard_size(image)
    image_service.to_bitmap(image_sized, render_config.output_path)

    return render_config.output_path.exists()
