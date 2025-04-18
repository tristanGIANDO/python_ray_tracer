from functools import reduce
from pathlib import Path

import numpy as np
from PIL import Image

from ray_tracer.application import Renderer
from ray_tracer.domain.models import (
    FARAWAY,
    Camera,
    Diffuse,
    DomeLight,
    Iridescence,
    Object3D,
    Scene3D,
    Shader,
    Specular,
)
from ray_tracer.domain.vector import Vector3D, extract


class Sphere3D(Object3D):
    def __init__(
        self,
        center: Vector3D,
        radius: float,
    ) -> None:
        self.center = center
        self.position = center
        self.radius = radius

    def intersect(self, ray_origin: Vector3D, normalized_ray_direction: Vector3D) -> np.ndarray:
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

    def get_uvs(self, hit_point: Vector3D) -> tuple[np.ndarray, np.ndarray]:
        normals = self.get_normals(hit_point)

        u = 0.5 + np.arctan2(normals.z, normals.x) / (2 * np.pi)
        v = 0.5 - np.arcsin(normals.y) / np.pi

        return u, v

    def get_normals(self, hit_point: Vector3D) -> Vector3D:
        return (hit_point - self.center) * (1.0 / self.radius)


class NumpyRenderer(Renderer):
    def raytrace_scene(
        self,
        ray_origin: Vector3D,
        normalized_ray_direction: Vector3D,
        scene: Scene3D,
    ) -> Vector3D:
        distances = [shape.intersect(ray_origin, normalized_ray_direction) for shape, _ in scene.shapes]
        nearest_distance = reduce(np.minimum, distances)  # noqa

        color = Vector3D(0, 0, 0)

        for shape_data, distance in zip(scene.shapes, distances, strict=False):
            hit = (nearest_distance != FARAWAY) & (distance == nearest_distance)

            if np.any(hit):  # keep only the intersected points
                extracted_distance_to_intersection: np.ndarray = extract(hit, distance)
                extracted_ray_origin: Vector3D = ray_origin.extract(hit)
                extracted_normalized_ray_direction: Vector3D = normalized_ray_direction.extract(hit)

                computed_color = shade(
                    shape_data,
                    scene,
                    extracted_ray_origin,
                    extracted_normalized_ray_direction,
                    extracted_distance_to_intersection,
                    self,
                    physical=True,
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

        return (Vector3D(x, y, 0) - camera.position).norm()

    def save_image(self, color: Vector3D, camera: Camera, output_path: str) -> None:
        rgb_colors = [
            Image.fromarray(
                (255 * np.clip(layer, 0, 1).reshape((camera.height, camera.width))).astype(np.uint8),
                "L",
            )
            for layer in color.components()
        ]
        Image.merge("RGB", rgb_colors).save(output_path)


def shade(
    shape_data: tuple[Object3D, Shader],
    scene: Scene3D,
    ray_origin: Vector3D,
    normalized_ray_direction: Vector3D,
    distance_origin_to_intersection: np.ndarray,
    ray_tracer: Renderer,
    physical: bool,
) -> Vector3D:
    """This is the main function that will be called to compute the color of the pixel."""
    shape, shader = shape_data
    hit_point = ray_origin + normalized_ray_direction * distance_origin_to_intersection
    normals = shape.get_normals(hit_point)
    direction_to_light = (scene.lights[0].position - hit_point).norm()
    direction_to_ray_origin = (scene.camera.position - hit_point).norm()
    nudged_intersection_point = hit_point + normals * 0.0001  # to avoid itself

    is_in_light = is_lightened(
        nudged_intersection_point,
        direction_to_light,
        scene,
        shape_data,
    )

    color = Vector3D(0.04, 0.04, 0.04)  # darkest ambient light

    if shader.diffuse is not None:
        color = calculate_diffuse(shader.diffuse, hit_point, normals, direction_to_light, is_in_light, shape)

    reflection = calculate_reflection(
        nudged_intersection_point,
        normals,
        normalized_ray_direction,
        scene,
        ray_tracer,
    )

    color += calculate_dome_light(normals, scene)

    if shader.specular is not None and physical:
        specular = calculate_physical_specular(
            shader.specular,
            normals,
            direction_to_light,
            direction_to_ray_origin,
        )

        color += (specular + reflection * 0.5) * shader.specular.intensity * is_in_light

    elif shader.specular is not None:
        specular = calculate_phong_specular(
            shader.specular,
            normals,
            direction_to_light,
            direction_to_ray_origin,
        )

        color += (specular + reflection * 0.5) * shader.specular.intensity * is_in_light

    if shader.iridescence is not None:
        color += calculate_iridescence(shader.iridescence, normals, direction_to_ray_origin)

    return color


def is_lightened(
    nudged_intersection_point: Vector3D,
    direction_to_light: Vector3D,
    scene: Scene3D,
    shape_data: tuple[Object3D, Shader],
) -> bool:
    """Calculates whether the intersection point is in light or in shadow.
    It does this by checking if the intersection point is closer to the light source.
    If the intersection point is closer to the light source than any other shape, it is in light.
    Otherwise, it is in shadow.
    """
    light_distances = [shape.intersect(nudged_intersection_point, direction_to_light) for shape, _ in scene.shapes]
    light_nearest = reduce(np.minimum, light_distances)
    return light_distances[scene.shapes.index(shape_data)] == light_nearest


def calculate_diffuse(
    diffuse: Diffuse,
    hit_point: Vector3D,
    normals: Vector3D,
    direction_to_light: Vector3D,
    is_in_light: bool,
    shape: Object3D,
) -> Vector3D:
    """Calculates the diffuse color of the shape at the intersection point."""
    diffuse_light_intensity = np.maximum(normals.dot(direction_to_light), 0)

    if diffuse.color and isinstance(diffuse.color, Path):
        diffuse_color = Image.open(diffuse.color).convert("RGB")

        u, v = shape.get_uvs(hit_point)
        diffuse_color = get_texture_color(diffuse_color, u, v)

    elif isinstance(diffuse.color, Vector3D):
        diffuse_color = diffuse.color
    else:
        diffuse_color = Vector3D(0.5, 0.5, 0.5)

    return diffuse_color * diffuse_light_intensity * is_in_light * diffuse.intensity


def calculate_reflection(  # TODO: VOIR TRANSMISSION
    nudged_intersection_point: Vector3D,
    normals: Vector3D,
    normalized_ray_direction: Vector3D,
    scene: Scene3D,
    ray_tracer: Renderer,
) -> Vector3D:
    ray_direction = (normalized_ray_direction - normals * 2 * normalized_ray_direction.dot(normals)).norm()
    color = ray_tracer.raytrace_scene(
        nudged_intersection_point,
        ray_direction,
        scene,
    )
    return Vector3D(
        np.array(color.x),
        np.array(color.y),
        np.array(color.z),
    )


def calculate_phong_specular(
    specular: Specular,
    normals: Vector3D,
    direction_to_light: Vector3D,
    direction_to_ray_origin: Vector3D,
) -> Vector3D:
    """Calculates the Phong specular reflection based on the angle between the normals and the direction
    to the light source.
    """
    phong = (direction_to_light + direction_to_ray_origin).norm()
    unclamped_roughness = (
        1 - specular.roughness
    ) * 100  # more user friendly to have roughness between 0 and 1 where 0 means no roughness and 1 means very rough

    return Vector3D(1, 1, 1) * np.power(np.clip(normals.dot(phong), 0, 1), unclamped_roughness) * self.specular_gain


def calculate_iridescence(  # TODO: NE SE VOIT QUE DANS LE SPECULAIRE!
    iridescence: Iridescence,
    normals: Vector3D,
    direction_to_ray_origin: Vector3D,
) -> Vector3D:
    """Calculates an iridescence color based on thin-film interference,
    incorporating three parameters:
        - thin_film_weight: the weight (coverage) that blends the film effect with the base,
        - thin_film_thickness: the thickness of the film, affecting the spacing of the fringes,
        - thin_film_ior: the refractive index of the film, influencing the hue of the fringes.

    The effect is achieved by modulating a sinusoidal oscillation (simulating interference)
    based on the angle between the normals and the direction to the ray origin.
    """
    # Calculate the viewing angle, in [0,1]
    view_angle = np.clip(normals.dot(direction_to_ray_origin), 0.0, 1.0)

    # An angular factor centered around 0.5 to enhance the effect at the edges
    angle_factor = np.abs(view_angle - 0.5) * 2.0

    # Use the thickness to adjust the frequency of the fringes:
    # Multiply by π and an arbitrary factor (here 10) to highlight oscillations.
    phase = angle_factor * np.pi * iridescence.thickness * 10.0

    # The interference pattern oscillates between -1 and 1
    interference_pattern = np.sin(phase)

    # The refractive index influences the hue. Here, thin_film_ior is mapped from [1,3] to a factor [0,1]
    # which will be used to vary the red and green components.
    hue_shift = (iridescence.ior - 1.0) / 2.0

    # Construct a fringe color whose intensity varies with the interference and the index:
    # - The red component is boosted when hue_shift is high,
    # - The green component is complemented inversely,
    # - The blue component is positioned around 0.5 and modulated by the interference.
    r = interference_pattern * hue_shift + (1.0 - hue_shift) * (1.0 - interference_pattern)
    g = interference_pattern * (1.0 - hue_shift) + hue_shift * (1.0 - interference_pattern)
    b = 0.5 + 0.5 * interference_pattern

    film_color = Vector3D(r, g, b)

    return film_color * iridescence.intensity


def calculate_dome_light(normals: Vector3D, scene: Scene3D) -> Vector3D:
    light_direction = Vector3D(0, 1, 0)  # from sky
    dome_intensity = 0.0
    dome_color = Vector3D(1.0, 1.0, 1.0)

    for light in scene.lights:
        if isinstance(light, DomeLight):
            dome_color = light.color
            dome_intensity += light.intensity * np.maximum(normals.dot(light_direction), 0)

    return dome_color * dome_intensity


def get_texture_color(texture: Image, u: np.ndarray, v: np.ndarray) -> Vector3D:
    texture_data = np.array(texture)  # shape = (height, width, channels)

    u = np.mod(u, 1)  # if u > 1, wrap around
    v = np.mod(v, 1)

    # find u and v in the image
    i = (u * (texture_data.shape[1] - 1)).astype(int)
    j = (v * (texture_data.shape[0] - 1)).astype(int)

    colors = texture_data[j, i, :3] / 255.0

    return Vector3D(colors[..., 0], colors[..., 1], colors[..., 2])


def calculate_physical_specular(
    specular: Specular,
    normals: Vector3D,
    direction_to_light: Vector3D,
    direction_to_ray_origin: Vector3D,
) -> Vector3D:
    """Calcule le terme spéculaire en combinant un modèle microfacettes GGX (avec Fresnel)
    et un boost de glint aux bords, visible uniquement si la lumière (domeLight ou équivalent)
    éclaire la surface.

    Le glint est modulé par l'intensité lumineuse incidente (via NdotL), de sorte que,
    en l'absence de lumière, il ne s'ajoute pas même si l'angle de vue est glancing.

    Args:
        specular (Specular): le matériau spéculaire.
        normals (Vector3D): la normale à la surface.
        direction_to_light (Vector3D): la direction (normalisée) vers la source de lumière.
        direction_to_ray_origin (Vector3D): la direction (normalisée) vers la caméra.

    Attributs attendus sur self:
        - self.specular_roughness: valeur dans [0, 1] (0 = surface parfaitement lisse).
        - self.specular_gain: gain appliqué au terme spéculaire.
        - self.specular_ior: indice de réfraction du matériau (pour Fresnel si non métallique).
        - self.is_metallic (optionnel): booléen indiquant si le matériau est métallique.
        - self.specular_color (optionnel): couleur de base du spéculaire pour les matériaux métalliques.
        - self.specular_glint_gain (optionnel): gain spécifique pour ajuster la contribution du glint.

    Returns:
        Vector3D: la couleur spéculaire résultante.
    """
    eps = 1e-8  # Petite constante pour éviter la division par zéro.

    # Normalisation des vecteurs
    L = direction_to_light.norm()  # Direction vers la lumière.
    V = direction_to_ray_origin.norm()  # Direction vers la caméra.
    H = (L + V).norm()  # Vecteur milieu.

    # Calcul des produits scalaires nécessaires
    NdotV = np.clip(normals.dot(V), 0, 1)
    NdotH = np.clip(normals.dot(H), 0, 1)
    VdotH = np.clip(V.dot(H), 0, 1)
    # Calcul de l'intensité lumineuse incidente (ici utilisée pour conditionner le glint)
    NdotL = np.clip(normals.dot(L), 0, 1)

    # ---- Fresnel (Schlick) ----
    F0 = ((specular.ior - 1) / (specular.ior + 1)) ** 2
    F = F0 + (1 - F0) * (1 - VdotH) ** 5

    # ---- Distribution GGX des microfacettes ----
    alpha = specular.roughness**2  # Le roughness au carré pour une réponse plus intuitive.
    denom = NdotH**2 * (alpha**2 - 1) + 1
    D = (alpha**2) / (np.pi * (denom**2 + eps))

    # ---- Terme Géométrique G (Smith Schlick-GGX) ----
    def G1(x: Vector3D) -> np.ndarray:
        xdotN = np.clip(normals.dot(x), 0, 1)
        return 2 * xdotN / (xdotN + np.sqrt(alpha**2 + (1 - alpha**2) * (xdotN**2)) + eps)

    G = G1(L) * G1(V)

    # ---- Composante spéculaire de base ----
    spec_base = (F * D * G) / (4 * NdotV + eps)

    # ---- Calcul du glint (edge highlight conditionné par la lumière) ----
    glint_exponent = 2.5  # Exposant ajustable pour moduler la décroissance du glint en fonction de l'angle.
    glint = (1 - NdotV) ** glint_exponent
    # On ne voit le glint que si la lumière l'éclaire, on le module donc par NdotL.
    glint *= NdotL

    # ---- Combinaison finale ----
    spec_final = spec_base + specular.intensity * glint

    # Masquage de la contribution si la surface n'est pas orientée vers la caméra.
    spec_final = Vector3D(1, 1, 1) * np.where(NdotV <= 0, 0, spec_final)

    return Vector3D(spec_final.x, spec_final.y, spec_final.z)
