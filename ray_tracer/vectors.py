import numpy as np


class Vector3D:
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

    def __add__(self, other: "Vector3D") -> "Vector3D":
        """
        Adds two vectors component-wise.

        Args:
            other (Vector3D): The vector to add.

        Returns:
            Vector3D: The resulting vector after addition.

        This method takes another vector and adds each component to the corresponding component of this vector.
        """
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        """
        Subtracts one vector from another component-wise.

        Args:
            other (Vector3D): The vector to subtract.

        Returns:
            Vector3D: The resulting vector after subtraction.

        This method subtracts the x, y, and z components of the other vector from this vector.
        """
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3D":
        """
        Multiplies the vector by a scalar or another vector component-wise.

        Args:
            scalar (float or Vector3D): The scalar or vector to multiply.

        Returns:
            Vector3D: The resulting vector after multiplication.

        If `scalar` is a float, each component of the vector is multiplied by this value.
        If `scalar` is another vector, each component is multiplied by the corresponding component of that vector.
        """
        if isinstance(scalar, Vector3D):  # Element-wise multiplication
            return Vector3D(self.x * scalar.x, self.y * scalar.y, self.z * scalar.z)
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> "Vector3D":
        """
        Divise le vecteur par un scalaire.

        Args:
            scalar (float): Le scalaire par lequel diviser.

        Returns:
            Vector3D: Un nouveau vecteur avec chaque composant divisé par le scalaire.
        """
        if scalar == 0:
            raise ValueError("Division par zéro non autorisée pour un vecteur.")
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: "Vector3D") -> float:
        """
        Computes the dot product of two vectors.

        Args:
            other (Vector3D): The vector to dot with.

        Returns:
            float: The dot product of the two vectors.

        The dot product is calculated as the sum of the products of the corresponding components:
        `x1 * x2 + y1 * y2 + z1 * z2`. This value represents how aligned two vectors are.
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> "Vector3D":
        """
        Normalizes the vector (scales it to have length 1).

        Returns:
            Vector3D: The normalized vector.

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

    def perturb(self, roughness: float) -> "Vector3D":
        """
        Perturbe ce vecteur en fonction de la roughness en utilisant un échantillonnage pondéré.

        Args:
            roughness (float): Facteur de rugosité (0 = lisse, 1 = rugueux).

        Returns:
            Vector3D: Vecteur perturbé en fonction de la rugosité.
        """
        if roughness <= 0:
            return self.norm()  # Aucune perturbation si la rugosité est nulle.

        # Générer un vecteur aléatoire dans un hémisphère autour de 'self'
        random_dir = Vector3D(
            np.random.normal(0, 1),
            np.random.normal(0, 1),
            np.random.normal(0, 1),
        ).norm()

        # Combiner la direction originale et la direction aléatoire selon la roughness
        perturbed_dir = self * (1 - roughness) + random_dir * roughness
        return perturbed_dir.norm()
