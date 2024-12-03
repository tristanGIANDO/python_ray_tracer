import numpy as np

from vector import Vec3


def random_unit_vector():
    theta = np.random.uniform(0, 2 * np.pi)
    z = np.random.uniform(0, 1)
    r = np.sqrt(1 - z**2)
    return Vec3(r * np.cos(theta), r * np.sin(theta), z)


def trace(ray_origin, ray_dir, scene, depth=0, max_depth=5):
    if depth >= max_depth:
        return Vec3(0, 0, 0)  # Pas de contribution au-delà de la profondeur maximale

    # Trouver l'objet le plus proche
    nearest_t, nearest_obj = float("inf"), None
    for obj in scene:
        t = obj.intersect(ray_origin, ray_dir)
        if t and t < nearest_t:
            nearest_t, nearest_obj = t, obj

    if nearest_obj is None:
        return Vec3(0, 0, 0)  # Couleur de fond (noir)

    hit_point = ray_origin + ray_dir * nearest_t
    normal = (hit_point - nearest_obj.center).norm()

    # Couleur directe (éclairage local)
    direct_color = nearest_obj.get_surface_color(hit_point)

    # Calcul Monte Carlo
    if depth < max_depth:
        # Générer un rayon aléatoire dans l'hémisphère
        random_dir = (normal + random_unit_vector()).norm()
        indirect_color = trace(hit_point + normal * 1e-4, random_dir, scene, depth + 1)
        return direct_color * 0.8 + indirect_color * 0.2  # Mélanger direct/indirect

    return direct_color


def render(scene, width, height, samples_per_pixel=10):
    aspect_ratio = float(width) / height
    camera = Vec3(0, 0, -1)
    screen = (-1, 1 / aspect_ratio, 1, -1 / aspect_ratio)

    image = np.zeros((height, width, 3))
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            color = Vec3(0, 0, 0)
            for _ in range(samples_per_pixel):
                # Ajouter une légère variation aléatoire à chaque rayon
                dx = np.random.uniform(-0.5, 0.5) / width
                dy = np.random.uniform(-0.5, 0.5) / height
                pixel = Vec3(x + dx, y + dy, 0)
                ray_dir = (pixel - camera).norm()
                color += trace(camera, ray_dir, scene)
            image[i, j] = np.clip(color.components(), 0, 1) / samples_per_pixel
    return image
