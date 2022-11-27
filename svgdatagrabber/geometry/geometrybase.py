from abc import ABC


class GeometryBase(ABC):
    #: Tolerance for equality.
    tolerance = 1e-6

    def __eq__(self, other: "GeometryBase") -> bool:
        """Check if two geometries are equal."""
        raise NotImplementedError

    def __ne__(self, other: "GeometryBase") -> bool:
        """Check if two geometries are not equal."""
        return not self.__eq__(other)
