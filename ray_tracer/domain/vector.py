import numbers
from typing import Any

import numpy as np


def extract(condition: bool, x: numbers.Number | int | float | np.ndarray) -> numbers.Number | np.ndarray:
    """Extracts elements from an array or returns the number itself if it is a scalar.

    Args:
        condition (array-like): Condition array to extract elements.
        x: Input number or array.
    """
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(condition, x)


class Vector3D:
    """Represents a 3D vector with numpy-based operations."""

    def __init__(self, x: int | float | np.ndarray, y: int | float | np.ndarray, z: int | float | np.ndarray) -> None:
        (self.x, self.y, self.z) = (x, y, z)

    def dot(self, other: "Vector3D") -> float | np.ndarray:
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self) -> float | np.ndarray:
        return self.dot(self)

    def __mul__(self, other: int | float) -> "Vector3D":
        if isinstance(other, Vector3D):
            return Vector3D(self.x * other.x, self.y * other.y, self.z * other.z)

        return Vector3D(self.x * other, self.y * other, self.z * other)

    def __add__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        """Defines the unary negation operator for Vector3D."""
        return Vector3D(-self.x, -self.y, -self.z)

    def __truediv__(self, other: int | float) -> "Vector3D":
        if isinstance(other, Vector3D):
            return Vector3D(self.x / other.x, self.y / other.y, self.z / other.z)
        return Vector3D(self.x / other, self.y / other, self.z / other)

    def components(self) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        return (self.x, self.y, self.z)

    def norm(self) -> Any:
        """Normalizes the vector to have a magnitude of 1."""
        mag = np.sqrt(self.dot(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def extract(self, condition: bool) -> "Vector3D | Vector3D":
        """Extracts components of the vector based on a condition."""
        if isinstance(condition, numbers.Number):
            return self

        return Vector3D(extract(condition, self.x), extract(condition, self.y), extract(condition, self.z))  # noqa

    def place(self, condition: np.ndarray) -> "Vector3D":
        """Places the vector's components into a new vector."""
        r = Vector3D(np.zeros(condition.shape), np.zeros(condition.shape), np.zeros(condition.shape))

        if not isinstance(r.x, np.ndarray):
            raise TypeError("r.x is not a numpy array")
        if not isinstance(r.y, np.ndarray):
            raise TypeError("r.y is not a numpy array")
        if not isinstance(r.z, np.ndarray):
            raise TypeError("r.z is not a numpy array")

        np.place(r.x, condition, self.x)
        np.place(r.y, condition, self.y)
        np.place(r.z, condition, self.z)
        return r
