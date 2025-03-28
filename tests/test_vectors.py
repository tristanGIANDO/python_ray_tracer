import numpy as np

from ray_tracer.domain.vectors import Vector3D


def test_vector_addition():
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    result = v1 + v2
    assert result.components() == (5, 7, 9)


def test_vector_subtraction():
    v1 = Vector3D(5, 7, 9)
    v2 = Vector3D(4, 5, 6)
    result = v1 - v2
    assert result.components() == (1, 2, 3)


def test_vector_dot_product():
    v1 = Vector3D(1, 0, 0)
    v2 = Vector3D(0, 1, 0)
    assert v1.dot(v2) == 0  # Perpendicular vectors

    v3 = Vector3D(1, 1, 1)
    assert v3.dot(v3) == 3  # Dot product with itself is its magnitude squared


def test_vector_normalization():
    v = Vector3D(3, 4, 0)
    norm = v.norm()
    assert np.isclose(norm.components(), (0.6, 0.8, 0)).all()


def test_vector_multiplication():
    v = Vector3D(1, 2, 3)
    result = v * 2
    assert result.components() == (2, 4, 6)


def test_vector_division():
    v = Vector3D(2, 4, 6)
    result = v / 2
    assert result.components() == (1, 2, 3)
