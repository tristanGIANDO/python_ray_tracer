from old_ray_tracer.domain import Light, Sphere
from old_ray_tracer.domain.vectors import Vector3D
from old_ray_tracer.ray_tracing import trace


def test_trace_no_intersection() -> None:
    scene = []
    lights = [Light(Vector3D(10, 10, -10), Vector3D(1, 1, 1))]
    color = trace(Vector3D(0, 0, -5), Vector3D(0, 0, 1).norm(), scene, lights)
    assert color.components() == (0, 0, 0)  # Background color


def test_trace_with_intersection() -> None:
    sphere = Sphere(Vector3D(0, 0, 0), 1, Vector3D(1, 0, 0), reflection=0.5)
    scene = [sphere]
    lights = [Light(Vector3D(10, 10, -10), Vector3D(1, 1, 1))]
    color = trace(Vector3D(0, 0, -5), Vector3D(0, 0, 1).norm(), scene, lights)
    assert color.components()[0] > 0  # The sphere is red


def test_trace_with_light_diffuse() -> None:
    sphere = Sphere(Vector3D(0, 0, 0), 1, Vector3D(1, 0, 0))
    light = Light(Vector3D(0, 10, -10), Vector3D(1, 1, 1))
    scene = [sphere]
    lights = [light]

    color = trace(Vector3D(0, 0, -5), Vector3D(0, 0, 1).norm(), scene, lights)
    assert color.components()[0] > 0  # Sphere is illuminated in red
    assert color.components()[1] == 0  # No green
    assert color.components()[2] == 0  # No blue


def test_roughness_low() -> None:
    sphere = Sphere(
        Vector3D(0, 0, 0), 1, Vector3D(1, 0, 0), reflection=0.8, roughness=0.2
    )
    light = Light(Vector3D(10, 10, -10), Vector3D(1, 1, 1))
    scene = [sphere]
    lights = [light]

    color = trace(Vector3D(0, 0, -5), Vector3D(0, 0, 1).norm(), scene, lights)
    assert color.components()[0] > 0  # Reflections are present
    assert color.components()[1] == 0  # No green
    assert color.components()[2] == 0  # No blue


def test_roughness_high() -> None:
    sphere = Sphere(
        Vector3D(0, 0, 0), 1, Vector3D(1, 0, 0), reflection=0.8, roughness=1.0
    )
    light = Light(Vector3D(10, 10, -10), Vector3D(1, 1, 1))
    scene = [sphere]
    lights = [light]

    color = trace(Vector3D(0, 0, -5), Vector3D(0, 0, 1).norm(), scene, lights)
    # Expect lower intensity due to diffused reflection
    assert color.components()[0] > 0  # Red should still be present


def test_combined_reflection_and_roughness() -> None:
    sphere = Sphere(
        Vector3D(0, 0, 0), 1, Vector3D(1, 0, 0), reflection=0.8, roughness=0.5
    )
    light = Light(Vector3D(10, 10, -10), Vector3D(1, 1, 1))
    scene = [sphere]
    lights = [light]

    color = trace(Vector3D(0, 0, -5), Vector3D(0, 0, 1).norm(), scene, lights)
    assert color.components()[0] > 0  # Red is present
    assert color.components()[1] == 0  # No green
    assert color.components()[2] == 0  # No blue


def test_roughness_statistical() -> None:
    sphere = Sphere(
        Vector3D(0, 0, 0), 1, Vector3D(1, 0, 0), reflection=0.8, roughness=0.8
    )
    light = Light(Vector3D(10, 10, -10), Vector3D(1, 1, 1))
    scene = [sphere]
    lights = [light]

    colors = []
    for _ in range(100):  # Test multiple traces to observe variation
        color = trace(Vector3D(0, 0, -5), Vector3D(0, 0, 1).norm(), scene, lights)
        colors.append(color.components()[0])  # Collect red channel

    assert max(colors) > 0.5  # Some reflections should be strong
    assert min(colors) < 0.5  # Some reflections should be weak
    assert len(set(colors)) > 10  # Ensure a range of results
