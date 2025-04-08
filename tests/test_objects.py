import pytest
from old_ray_tracer.domain import Sphere
from old_ray_tracer.domain.vectors import Vector3D


def test_sphere_intersection() -> None:
    sphere = Sphere(Vector3D(0, 0, 0), 1, Vector3D(1, 0, 0))
    ray_origin = Vector3D(0, 0, -3)
    ray_dir = Vector3D(0, 0, 1).norm()
    t = sphere.intersect(ray_origin, ray_dir)
    assert t == pytest.approx(2.0)


def test_sphere_no_intersection() -> None:
    sphere = Sphere(Vector3D(0, 0, 0), 1, Vector3D(1, 0, 0))
    ray_origin = Vector3D(0, 0, -3)
    ray_dir = Vector3D(0, 1, 1).norm()
    t = sphere.intersect(ray_origin, ray_dir)
    assert t is None


def test_sphere_get_surface_color() -> None:
    sphere = Sphere(Vector3D(0, 0, 0), 1, Vector3D(1, 0, 0))
    color = sphere.get_surface_color(Vector3D(1, 0, 0))
    assert color.components() == (1, 0, 0)
