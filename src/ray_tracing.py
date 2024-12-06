import numpy as np
from PIL import Image


class Vec3:
    def __init__(self, x: float, y: float, z: float) -> None:
        """
        Initializes a 3D vector.

        Args:
            x (float): X component of the vector.
            y (float): Y component of the vector.
            z (float): Z component of the vector.

        This class is used to represent a point or direction in 3D space. Each vector has three components: x, y, and z.
        """
        self.x, self.y, self.z = x, y, z

    def __add__(self, other: "Vec3") -> "Vec3":
        """
        Adds two vectors component-wise.

        Args:
            other (Vec3): The vector to add.

        Returns:
            Vec3: The resulting vector after addition.

        This method takes another vector and adds each component to the corresponding component of this vector.
        """
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        """
        Subtracts one vector from another component-wise.

        Args:
            other (Vec3): The vector to subtract.

        Returns:
            Vec3: The resulting vector after subtraction.

        This method subtracts the x, y, and z components of the other vector from this vector.
        """
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float | "Vec3") -> "Vec3":
        """
        Multiplies the vector by a scalar or another vector component-wise.

        Args:
            scalar (float or Vec3): The scalar or vector to multiply.

        Returns:
            Vec3: The resulting vector after multiplication.

        If `scalar` is a float, each component of the vector is multiplied by this value.
        If `scalar` is another vector, each component is multiplied by the corresponding component of that vector.
        """
        if isinstance(scalar, Vec3):  # Element-wise multiplication
            return Vec3(self.x * scalar.x, self.y * scalar.y, self.z * scalar.z)
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: "Vec3") -> float:
        """
        Computes the dot product of two vectors.

        Args:
            other (Vec3): The vector to dot with.

        Returns:
            float: The dot product of the two vectors.

        The dot product is calculated as the sum of the products of the corresponding components:
        `x1 * x2 + y1 * y2 + z1 * z2`. This value represents how aligned two vectors are.
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> "Vec3":
        """
        Normalizes the vector (scales it to have length 1).

        Returns:
            Vec3: The normalized vector.

        This method calculates the length of the vector and scales all components so that the vector has a length of 1.
        This is useful for directions. If the length is very small (close to zero), a small value is used to prevent division by zero.
        """
        length = np.sqrt(self.dot(self))
        return self * (1.0 / max(length, 1e-6))

    def components(self) -> tuple[float, float, float]:
        """
        Returns the components of the vector as a tuple.

        Returns:
            tuple: A tuple containing the x, y, and z components of the vector.

        This method is used to easily extract all three components of the vector for further calculations or storage.
        """
        return self.x, self.y, self.z

class Light:
    def __init__(self, position: Vec3, intensity: Vec3) -> None:
        """
        Initializes a light source.

        Args:
            position (Vec3): The position of the light source.
            intensity (Vec3): The intensity and color of the light.

        The position determines where the light is located in the scene, and the intensity determines how bright the light is and its color.
        """
        self.position = position
        self.intensity = intensity

def get_texture_color(texture: Image, u: float, v: float) -> Vec3:
    """
    Retrieves the color from a texture image at given UV coordinates.

    Args:
        texture (Image): The texture image.
        u (float): U coordinate (horizontal).
        v (float): V coordinate (vertical).

    Returns:
        Vec3: The color at the specified UV coordinates, normalized between 0 and 1.

    The UV coordinates (u, v) are like X and Y positions on the texture image. This function maps those coordinates to the
    actual pixel in the image, allowing the color to be sampled. The colors are then normalized to be between 0 and 1.
    """
    texture_data = np.array(texture)
    u = u % 1  # Repeat if u exceeds 1
    v = v % 1
    i = int(u * (texture_data.shape[1] - 1))
    j = int(v * (texture_data.shape[0] - 1))
    return Vec3(*texture_data[j, i, :3] / 255)

class Sphere:
    def __init__(self, center: Vec3, radius: Vec3, color: Vec3, reflection:float|None=0.5, texture=None) -> None:
        """
        Initializes a sphere object.

        Args:
            center (Vec3): The center position of the sphere.
            radius (float): The radius of the sphere.
            color (Vec3): The base color of the sphere.
            reflection (float, optional): Reflection coefficient of the sphere. Defaults to 0.5.
            texture (Image, optional): Texture image for the sphere. Defaults to None.

        A sphere is defined by its position, size, color, and optionally, a texture and reflection properties. The texture
        allows the sphere to have more complex surface details.
        """
        self.center = center
        self.radius = radius
        self.color = color
        self.reflection = reflection
        self.texture = texture

    def intersect(self, ray_origin: Vec3, ray_dir: Vec3) -> float|None:
        """
        Computes the intersection of a ray with the sphere.

        Args:
            ray_origin (Vec3): The origin of the ray.
            ray_dir (Vec3): The direction of the ray.

        Returns:
            float or None: The distance from the ray origin to the intersection point, or None if no intersection.

        This method uses the mathematical formula for the intersection of a ray and a sphere.
        It solves a quadratic equation to find where (if at all) the ray hits the sphere.
        The discriminant (`b^2 - 4 * c`) determines if there are intersections (real solutions).
        If the discriminant is positive, there are two possible intersection points.
        """
        oc = ray_origin - self.center
        b = 2.0 * oc.dot(ray_dir)
        c = oc.dot(oc) - self.radius ** 2
        disc = b ** 2 - 4 * c
        if disc > 0:
            sqrtd = np.sqrt(disc)
            t0 = (-b - sqrtd) / 2
            t1 = (-b + sqrtd) / 2
            if t0 > 0:
                return t0
            if t1 > 0:
                return t1
        return None

    def get_surface_color(self, hit_point: Vec3) -> Vec3:
        """
        Gets the color of the surface at a given hit point.

        Args:
            hit_point (Vec3): The point on the surface of the sphere.

        Returns:
            Vec3: The color at the hit point.

        If the sphere has a texture, this method calculates texture coordinates (u, v) based on the position of the hit point.
        It then uses these coordinates to look up the color from the texture image. If no texture is present, it returns the base color.
        """
        try:
            if self.texture.any():
                normal = (hit_point - self.center).norm()
                u = 0.5 + np.arctan2(normal.z, normal.x) / (2 * np.pi)
                v = 0.5 - np.arcsin(normal.y) / np.pi
                return get_texture_color(self.texture, u, v)
        except AttributeError:
            return self.color

def trace(ray_origin: Vec3, ray_dir: Vec3, scene: list[Sphere], lights: list[Light]) -> Vec3:
    """
    Traces a ray through the scene to determine the color at a given point.

    Args:
        ray_origin (Vec3): The origin of the ray.
        ray_dir (Vec3): The direction of the ray.
        scene (list): A list of objects in the scene.
        lights (list): A list of light sources in the scene.

    Returns:
        Vec3: The color determined by tracing the ray.

    This function finds the closest object that the ray intersects. It calculates where the ray hits the object, then determines
    the color at that point based on the object's properties and the lighting conditions. It also checks for shadows by sending
    rays toward each light source to see if they are blocked.
    """
    nearest_t, nearest_obj = float('inf'), None
    for obj in scene:
        t = obj.intersect(ray_origin, ray_dir)
        if t and t < nearest_t:
            nearest_t, nearest_obj = t, obj

    if nearest_obj is None:
        return Vec3(0, 0, 0)  # Background color (black)

    hit_point = ray_origin + ray_dir * nearest_t
    normal = (hit_point - nearest_obj.center).norm() if isinstance(nearest_obj, Sphere) else nearest_obj.normal
    color = nearest_obj.get_surface_color(hit_point)

    light_contribution = Vec3(0, 0, 0)
    for light in lights:
        light_dir = (light.position - hit_point).norm()
        shadow_ray_origin = hit_point + normal * 1e-4
        shadow_intersect = any(
            obj.intersect(shadow_ray_origin, light_dir) for obj in scene if obj != nearest_obj
        )
        if not shadow_intersect:
            intensity = max(normal.dot(light_dir), 0)
            light_contribution += light.intensity * intensity

    return color * light_contribution

def render(scene: list[Sphere], lights: list[Light], width: int, height: int) -> np.ndarray:
    """
    Renders the scene to create an image.

    Args:
        scene (list): A list of objects in the scene.
        lights (list): A list of light sources in the scene.
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        numpy.ndarray: The rendered image as an array of pixel values.

    This function represents the camera looking at the scene through a grid of pixels (the image).
    For each pixel, it sends a ray from the camera into the scene to determine what color that pixel should be.
    It calculates the direction of each ray and uses the `trace` function to determine the color based on object interactions.
    """
    aspect_ratio = float(width) / height
    camera = Vec3(0, 0, -1)
    screen = (-1, 1 / aspect_ratio, 1, -1 / aspect_ratio)

    image = np.zeros((height, width, 3))
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = Vec3(x, y, 0)
            ray_dir = (pixel - camera).norm()
            color = trace(camera, ray_dir, scene, lights)
            image[i, j] = np.clip(color.components(), 0, 1)

    return image


if __name__ == "__main__":
    # Load a texture for the plane
    floor_texture = np.array(Image.open("resource/2k_earth_daymap.jpg"))

    # Define objects in the scene
    scene = [
        Sphere(Vec3(0, -0.5, 3), 0.5, Vec3(1, 0, 0)),
        Sphere(Vec3(1, 0, 4), 1, Vec3(0, 1, 0)),
        Sphere(Vec3(-1, 0, 2.5), 0.3, Vec3(0, 0, 1)),
    ]

    # Define lights
    lights = [
        Light(Vec3(5, 10, -10), Vec3(1, 1, 1))  # Point white light
    ]

    # Render the image
    width, height = 500, 500
    image = render(scene, lights, width, height)
    Image.fromarray((image * 255).astype(np.uint8)).save("final_render.png")
