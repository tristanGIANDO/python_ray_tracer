from typing import Generator

import numpy as np

from ray_tracer.domain import HDRI, RenderConfig, Scene, Sphere, Vector3D
from ray_tracer.infrastructure.pillow import open_image
from ray_tracer.services import ObjectService, RayTracer, VectorService


def get_texture_color(texture: Image, u: float, v: float) -> Vector3D:
    """
    Retrieves the color from a texture image at given UV coordinates.

    Args:
        texture (Image): The texture image.
        u (float): U coordinate (horizontal).
        v (float): V coordinate (vertical).

    Returns:
        Vector3D: The color at the specified UV coordinates, normalized between 0 and 1.

    The UV coordinates (u, v) are like X and Y positions on the texture image. This function maps those coordinates to the
    actual pixel in the image, allowing the color to be sampled. The colors are then normalized to be between 0 and 1.
    """
    texture_data = np.array(texture)
    u = u % 1  # Repeat if u exceeds 1
    v = v % 1
    i = int(u * (texture_data.shape[1] - 1))
    j = int(v * (texture_data.shape[0] - 1))
    return Vector3D(*texture_data[j, i, :3] / 255)


class NumpySphereService(ObjectService):
    def __init__(self, sphere: Sphere) -> None:
        self.sphere = sphere

    def intersect(self, ray_origin: Vector3D, ray_dir: Vector3D) -> float | None:
        """
        Computes the intersection of a ray with the sphere.

        Args:
            ray_origin (Vector3D): The origin of the ray.
            ray_dir (Vector3D): The direction of the ray.

        Returns:
            float or None: The distance from the ray origin to the intersection point, or None if no intersection.

        This method uses the mathematical formula for the intersection of a ray and a sphere.
        It solves a quadratic equation to find where (if at all) the ray hits the sphere.
        The discriminant (`b^2 - 4 * c`) determines if there are intersections (real solutions).
        If the discriminant is positive, there are two possible intersection points.
        """
        oc = ray_origin - self.sphere.center
        b = 2.0 * oc.dot(ray_dir)
        c = oc.dot(oc) - self.sphere.radius**2
        disc = b**2 - 4 * c
        if disc > 0:
            sqrtd = np.sqrt(disc)
            t0 = (-b - sqrtd) / 2
            t1 = (-b + sqrtd) / 2
            if t0 > 0:
                return t0
            if t1 > 0:
                return t1
        return None

    def get_surface_color(self, hit_point: Vector3D) -> Vector3D:
        """
        Gets the color of the surface at a given hit point.

        Args:
            hit_point (Vector3D): The point on the surface of the sphere.

        Returns:
            Vector3D: The color at the hit point.

        If the sphere has a texture, this method calculates texture coordinates (u, v) based on the position of the hit point.
        It then uses these coordinates to look up the color from the texture image. If no texture is present, it returns the base color.
        """
        try:
            if self.sphere.texture.any():
                normal = (hit_point - self.sphere.center).norm()
                u = 0.5 + np.arctan2(normal.z, normal.x) / (2 * np.pi)
                v = 0.5 - np.arcsin(normal.y) / np.pi
                return get_texture_color(self.sphere.texture, u, v)
        except AttributeError:
            return self.sphere.color


class NumpyVectorService(VectorService):
    def add(self, vector1: Vector3D, vector2: Vector3D) -> Vector3D:
        """
        Adds two vectors component-wise.

        Args:
            vector1 (Vector3D): The first vector.
            vector2 (Vector3D): The second vector.

        Returns:
            Vector3D: The resulting vector after addition.

        This method takes another vector and adds each component to the corresponding component of this vector.
        """
        return Vector3D(
            vector1.x + vector2.x, vector1.y + vector2.y, vector1.z + vector2.z
        )

    def sub(self, vector1: Vector3D, vector2: Vector3D) -> Vector3D:
        """
        Subtracts one vector from another component-wise.

        Args:
            vector1 (Vector3D): The first vector.
            vector2 (Vector3D): The second vector.

        Returns:
            Vector3D: The resulting vector after subtraction.

        This method subtracts the x, y, and z components of the other vector from this vector.
        """
        return Vector3D(
            vector1.x - vector2.x, vector1.y - vector2.y, vector1.z - vector2.z
        )

    def mul(self, vector: Vector3D, scalar: float | Vector3D) -> Vector3D:
        """
        Multiplies the vector by a scalar or another vector component-wise.

        Args:
            vector (Vector3D): The vector to multiply.
            scalar (float or Vector3D): The scalar or vector to multiply.

        Returns:
            Vector3D: The resulting vector after multiplication.

        If `scalar` is a float, each component of the vector is multiplied by this value.
        If `scalar` is another vector, each component is multiplied by the corresponding component of that vector.
        """
        if isinstance(scalar, Vector3D):  # Element-wise multiplication
            return Vector3D(
                vector.x * scalar.x, vector.y * scalar.y, vector.z * scalar.z
            )
        return Vector3D(vector.x * scalar, vector.y * scalar, vector.z * scalar)

    def truediv(self, vector: Vector3D, scalar: float) -> Vector3D:
        """
        Divides the vector by a scalar.

        Args:
            vector (Vector3D): The vector to divide.
            scalar (float): The scalar to divide by.

        Returns:
            Vector3D: The resulting vector after division.

        This method divides each component of the vector by the scalar value.
        """
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return Vector3D(vector.x / scalar, vector.y / scalar, vector.z / scalar)

    def dot(self, vector1: Vector3D, vector2: Vector3D) -> float:
        """
        Computes the dot product of two vectors.

        Args:
            vector1 (Vector3D): The first vector.
            vector2 (Vector3D): The second vector.

        Returns:
            float: The dot product of the two vectors.

        The dot product is the sum of the products of the corresponding components of the two vectors.
        This value represents how aligned two vectors are.
        """
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

    def normalize(self, vector: Vector3D) -> float:
        length = np.sqrt(self.dot(vector, vector))
        return self.truediv(vector, max(length, 1e-6))

    def components(self, vector: Vector3D) -> tuple[float, float, float]:
        return vector.x, vector.y, vector.z

    def perturb(self, vector: Vector3D, scale: float) -> Vector3D:
        """
        Perturbs the vector by a random amount.

        Args:
            vector (Vector3D): The vector to perturb.
            scale (float): The maximum amount to perturb by.

        Returns:
            Vector3D: The perturbed vector.

        This method adds a random vector to the input vector, scaled by the given amount.
        """
        if scale <= 0:
            return self.normalize(vector)  # no perturbation if scale is zero

        # Générer un vecteur aléatoire dans un hémisphère autour de 'self'
        random_dir = Vector3D(
            np.random.normal(0, 1),
            np.random.normal(0, 1),
            np.random.normal(0, 1),
        ).norm()

        # Combiner la direction originale et la direction aléatoire selon la roughness
        perturbed_dir = vector * (1 - scale) + random_dir * scale
        return perturbed_dir.norm()


def get_hdri_color(hdri: HDRI, direction: Vector3D) -> Vector3D:
    """
    Récupère la couleur de l'environnement en fonction de la direction du rayon.

    Args:
        direction (Vector3D): La direction du rayon.

    Returns:
        Vector3D: La couleur de l'environnement.
    """
    image_data = np.array(open_image(hdri)) / 255.0  # Normaliser entre 0 et 1
    width = image_data.shape[1]
    height = image_data.shape[0]

    # Convertir le vecteur en coordonnées sphériques
    theta = np.arccos(direction.y)  # Angle de l'axe Y
    phi = np.arctan2(direction.z, direction.x)  # Angle autour de l'axe Y

    # Normaliser phi pour qu'il soit entre [0, 1]
    u = (phi + np.pi) / (2 * np.pi)
    v = theta / np.pi

    # Convertir (u, v) en coordonnées d'image
    i = int(u * (width - 1))
    j = int((1 - v) * (height - 1))

    # Récupérer la couleur et retourner comme vecteur
    color = image_data[j, i, :3]
    return Vector3D(*color)


class NumpyRayTracer(RayTracer):
    def _trace_ray(
        self,
        ray_origin: Vector3D,
        ray_dir: Vector3D,
        scene: list[Sphere],
        depth: int = 0,
        max_depth: int = 3,
    ) -> Vector3D:
        """
        Traces a ray through the scene to determine the color at a given point, including reflections.

        Args:
            ray_origin (Vector3D): The origin of the ray.
            ray_dir (Vector3D): The direction of the ray.
            scene (list): A list of objects in the scene.
            lights (list): A list of light sources in the scene.
            depth (int): Current recursion depth for reflections.
            max_depth (int): Maximum recursion depth for reflections.

        Returns:
            Vector3D: The color determined by tracing the ray.
        """
        nearest_t, nearest_obj = float("inf"), None
        for obj in scene.objects:
            t = obj.intersect(ray_origin, ray_dir)
            if t and t < nearest_t:
                nearest_t, nearest_obj = t, obj

        if nearest_obj is None:
            # Retourner la couleur de l'environnement si aucune intersection
            if scene.background:
                return scene.background.get_color(ray_dir)
            return Vector3D(0, 0, 0)  # Couleur de fond noire

        # Point d'intersection
        hit_point = ray_origin + ray_dir * nearest_t
        normal = (hit_point - nearest_obj.center).norm()
        color = nearest_obj.get_surface_color(hit_point)

        # Calculate direct lighting (diffuse shading)
        light_contribution = Vector3D(0, 0, 0)
        for light in scene.lights:
            light_dir = (light.position - hit_point).norm()
            shadow_ray_origin = hit_point + normal * 1e-4
            shadow_intersect = any(
                obj.intersect(shadow_ray_origin, light_dir)
                for obj in scene.objects
                if obj != nearest_obj
            )
            if not shadow_intersect:
                intensity = max(normal.dot(light_dir), 0)
                light_contribution += light.intensity * intensity

        # Reflection
        reflection_contribution = Vector3D(0, 0, 0)
        if (
            depth < max_depth
            and hasattr(nearest_obj, "reflection")  # TODO: meta programming !!
            and nearest_obj.reflection > 0
        ):
            reflected_dir = (ray_dir - normal * (2 * normal.dot(ray_dir))).norm()

            # Appliquer la roughness si elle est définie
            if hasattr(nearest_obj, "roughness") and nearest_obj.roughness > 0:
                reflected_dir = reflected_dir.perturb(nearest_obj.roughness)

            reflected_origin = hit_point + normal * 1e-4
            reflection_color = self._trace_ray(
                reflected_origin, reflected_dir, scene, depth + 1, max_depth
            )
            reflection_contribution = reflection_color * nearest_obj.reflection

        # Combiner les contributions
        return color * light_contribution + reflection_contribution

    def render(
        self,
        scene: Scene,
        render_config: RenderConfig,  # MonteCarlo live
    ) -> Generator[np.ndarray, np.ndarray, np.ndarray]:
        aspect_ratio = float(render_config.image_width) / render_config.image_height
        camera = Vector3D(0, 0, -1)
        screen = (-1, 1 / aspect_ratio, 1, -1 / aspect_ratio)

        image = np.zeros((render_config.image_height, render_config.image_width, 3))
        accumulated_color = np.zeros(
            (render_config.image_height, render_config.image_width, 3)
        )

        for sample in range(1, render_config.samples_per_pixel + 1):
            for i, y in enumerate(
                np.linspace(screen[1], screen[3], render_config.image_height)
            ):
                for j, x in enumerate(
                    np.linspace(screen[0], screen[2], render_config.image_width)
                ):
                    # Ajouter une petite variation aléatoire pour chaque rayon
                    u = np.random.uniform(
                        -1 / render_config.image_width, 1 / render_config.image_width
                    )
                    v = np.random.uniform(
                        -1 / render_config.image_height, 1 / render_config.image_height
                    )
                    pixel = Vector3D(x + u, y + v, 0)
                    ray_dir = (pixel - camera).norm()
                    color = self._trace_ray(camera, ray_dir, scene)
                    accumulated_color[i, j] += color.components()

            # Mettre à jour l'image avec les moyennes actuelles
            image = np.clip(accumulated_color / sample, 0, 1)
            yield image

    def to_standard_size(self, image: np.ndarray) -> np.ndarray:
        return (image * 255).astype(np.uint8)
