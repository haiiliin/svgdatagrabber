from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Iterable, Union, Sequence, overload, List

import numpy as np

from .geometrybase import GeometryBase

PointType = Union["Point", Iterable[float], complex]


class Point(GeometryBase):
    #: The x coordinate of the point.
    x: float
    #: The y coordinate of the point.
    y: float

    def __init__(self, *args: float | PointType, x: float = None, y: float = None):
        """Initialize a point.

        >>> Point(1.0, 2.0)
        Point(x=1.0, y=2.0)
        >>> Point([1.0, 2.0])
        Point(x=1.0, y=2.0)
        >>> Point(complex(1.0, 2.0))
        Point(x=1.0, y=2.0)
        >>> Point(Point(1.0, 2.0))
        Point(x=1.0, y=2.0)
        >>> Point(x=1.0, y=2.0)
        Point(x=1.0, y=2.0)
        >>> Point(1.0)
        Traceback (most recent call last):
        ...
        ValueError: The arguments must be two floats, a complex number or an iterable of two floats.

        Args:
            args: two floats, a complex number or an iterable of two floats.
        """
        if len(args) == 1 and isinstance(args[0], complex):
            self.x, self.y = args[0].real, args[0].imag
        elif len(args) == 1 and isinstance(args[0], Iterable) and len(tuple(args[0])) == 2:
            self.x, self.y = args[0]
        elif len(args) == 2:
            self.x, self.y = args
        elif len(args) == 0 and x is not None and y is not None:
            self.x, self.y = x, y
        else:
            raise ValueError("The arguments must be two floats, a complex number or an iterable of two floats.")

    def __repr__(self) -> str:
        """Return a string representation of the point.

        >>> repr(Point(1.0, 2.0))
        'Point(x=1.0, y=2.0)'
        """
        name = self.__class__.__name__
        x, y = round(self.x, 10), round(self.y, 10)
        return f"{name}(x={x}, y={y})"

    def __iter__(self):
        """Iterate over the coordinates of the point.

        >>> point = Point(1.0, 2.0)
        >>> x, y = point
        >>> assert x == 1.0
        >>> assert y == 2.0
        """
        yield self.x
        yield self.y

    def __neg__(self) -> Point:
        """Return the negative of the point.

        >>> point = -Point(1.0, 2.0)
        >>> assert point.x == -1.0
        >>> assert point.y == -2.0
        """
        return self.__class__(-self.x, -self.y)

    def __eq__(self, other: PointType) -> bool:
        """Check if two points are equal.

        >>> Point(1.0, 2.0) == Point(1.0, 2.0)
        True
        >>> Point(1.0, 2.0) != Point(1.0, 2.0)
        False
        >>> Point(1.0, 2.0) == Point(1.0, 3.0)
        False
        >>> Point(1.0, 2.0) != Point(1.0, 3.0)
        True

        Args:
            other: The other point.
        """
        other = self.aspoint(other)
        return np.allclose([self.x, self.y], [other.x, other.y], atol=self.tolerance)

    def __add__(self, other: PointType) -> Point:
        """Add a point or vector to the point.

        >>> Point(1.0, 2.0) + Point(3.0, 4.0)
        Point(x=4.0, y=6.0)

        Args:
            other: The other point.
        """
        other = self.aspoint(other)
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: PointType) -> Point:
        """Subtract a point or vector from the point.

        >>> Point(1.0, 2.0) - Point(3.0, 4.0)
        Point(x=-2.0, y=-2.0)

        Args:
            other: The other point.
        """
        other = self.aspoint(other)
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float) -> Point:
        """Multiply the point by a scalar.

        >>> Point(1.0, 2.0) * 2.0
        Point(x=2.0, y=4.0)

        Args:
            other: A scalar value.
        """
        return Point(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> Point:
        """Divide the point by a scalar.

        >>> Point(1.0, 2.0) / 2.0
        Point(x=0.5, y=1.0)

        Args:
            other: A scalar value.
        """
        return Point(self.x / other, self.y / other)

    def __abs__(self) -> float:
        """Calculate the magnitude of the vector.

        >>> abs(Point(3.0, 4.0))
        5.0

        Returns:
            The magnitude of the vector.
        """
        return np.linalg.norm(self.array)

    @classmethod
    def aspoint(cls, p: PointType) -> Point:
        """Convert a point to a Point object.

        >>> Point.aspoint(Point(1.0, 2.0))
        Point(x=1.0, y=2.0)
        >>> Point.aspoint((1.0, 2.0))
        Point(x=1.0, y=2.0)
        >>> Point.aspoint(1.0 + 2.0j)
        Point(x=1.0, y=2.0)
        >>> Point.aspoint(1.0)
        Traceback (most recent call last):
        ...
        TypeError: Cannot convert 1.0 to a Point object.

        Args:
            p: The point to convert.

        Returns:
            The converted point.
        """
        if isinstance(p, cls):
            return p
        elif isinstance(p, complex):
            return cls(p.real, p.imag)
        elif isinstance(p, Iterable):
            return cls(*tuple(p))
        else:
            raise TypeError(f"Cannot convert {p} to a {cls.__name__} object.")

    def distance(self, other: PointType) -> float:
        """Calculate the distance between two points.

        >>> Point(1.0, 2.0).distance(Point(4.0, 6.0))
        5.0

        Args:
            other: The other point.

        Returns:
            The distance between the points.
        """
        other = self.aspoint(other)
        return np.linalg.norm(np.array([self.x, self.y]) - np.array([other.x, other.y]))

    def direction(self, other: PointType) -> float:
        """Calculate the direction between two points.

        >>> p1 = Point(0.0, 0.0).direction(Point(1.0, 1.0))
        >>> assert np.isclose(p1, np.pi / 4.0)
        >>> p2 = Point(0.0, 0.0).direction(Point(-1.0, 1.0))
        >>> assert np.isclose(p2, 3.0 * np.pi / 4.0)
        >>> p2 = Point(0.0, 0.0).direction(Point(-1.0, -1.0))
        >>> assert np.isclose(p2, -3.0 * np.pi / 4.0)
        >>> p2 = Point(0.0, 0.0).direction(Point(1.0, -1.0))
        >>> assert np.isclose(p2, -np.pi / 4.0)

        Args:
            other: The other point.

        Returns:
            The direction between the points.
        """
        other = self.aspoint(other)
        return np.arctan2(other.y - self.y, other.x - self.x)

    def vector(self, other: PointType) -> Vector:
        """Calculate the vector between two points.

        >>> Point(1.0, 2.0).vector(Point(4.0, 6.0))
        Vector(x=3.0, y=4.0)

        Args:
            other: The other point.

        Returns:
            The vector between the points.
        """
        other = self.aspoint(other)
        return Vector(other.x - self.x, other.y - self.y)

    @property
    def array(self) -> np.ndarray:
        """Convert the vector to a numpy array.

        >>> Point(1.0, 2.0).array
        array([1., 2.])

        Returns:
            The vector as a numpy array.
        """
        return np.array([self.x, self.y])


class Vector(Point):
    def __matmul__(self, other: PointType) -> float:
        """Calculate the dot product between two points.

        >>> Vector(1.0, 2.0) @ Vector(3.0, 4.0)
        11.0

        Args:
            other: The other point.

        Returns:
            The dot product between the points.
        """
        other = self.aspoint(other)
        return self.x * other.x + self.y * other.y

    def dot(self, other: PointType) -> float:
        """Calculate the dot product between two points.

        >>> Vector(1.0, 2.0).dot(Vector(3.0, 4.0))
        11.0

        Args:
            other: The other point.

        Returns:
            The dot product between the points.
        """
        return self.__matmul__(other)

    @classmethod
    def asvector(cls, v: "Vector" | PointType) -> "Vector":
        """Convert a vector to a Vector object.

        >>> Vector.asvector(Vector(1.0, 2.0))
        Vector(x=1.0, y=2.0)
        >>> Vector.asvector((1.0, 2.0))
        Vector(x=1.0, y=2.0)
        >>> Vector.asvector(1.0 + 2.0j)
        Vector(x=1.0, y=2.0)

        Args:
            v: The vector to convert.

        Returns:
            The converted vector.
        """
        return cls.aspoint(v)


class IterablePoint(ABC):
    def __iter__(self) -> Iterable[Point]:
        """Iterate over the geometry vertices."""
        raise NotImplementedError


class PointSequence(Sequence[Point], GeometryBase, IterablePoint):
    #: The points in the sequence.
    points: List[Point]

    def __init__(self, *points: PointType | Iterable[PointType]):
        """Create a sequence of points.

        >>> PointSequence(Point(1.0, 2.0), Point(3.0, 4.0))
        PointSequence(Point(x=1.0, y=2.0), Point(x=3.0, y=4.0))
        >>> PointSequence([(1.0, 2.0), (3.0, 4.0)])
        PointSequence(Point(x=1.0, y=2.0), Point(x=3.0, y=4.0))

        Args:
            points: The points to add to the sequence.
        """
        if len(points) == 1 and isinstance(points[0], Iterable):
            points = points[0]
        self.points = [Point.aspoint(p) for p in points]

    def __repr__(self) -> str:
        """Return a string representation of the sequence.

        >>> repr(PointSequence(Point(1.0, 2.0), Point(3.0, 4.0)))
        'PointSequence(Point(x=1.0, y=2.0), Point(x=3.0, y=4.0))'
        """
        return f"{self.__class__.__name__}({', '.join(repr(p) for p in self.points)})"

    @overload
    @abstractmethod
    def __getitem__(self, index: int) -> Point:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, index: slice) -> Sequence[Point]:
        ...

    def __getitem__(self, index: int) -> Point | Sequence[Point]:
        """Get a point from the sequence.

        >>> PointSequence(Point(1.0, 2.0), Point(3.0, 4.0))[0]
        Point(x=1.0, y=2.0)
        >>> PointSequence(Point(1.0, 2.0), Point(3.0, 4.0))[1]
        Point(x=3.0, y=4.0)
        >>> PointSequence(Point(1.0, 2.0), Point(3.0, 4.0))[0:1]
        [Point(x=1.0, y=2.0)]

        Args:
            index: The index of the point to get.
        """
        return self.points[index]

    def __len__(self) -> int:
        """Return the length of the sequence.

        >>> len(PointSequence(Point(1.0, 2.0), Point(3.0, 4.0)))
        2
        """
        return len(self.points)

    def __iter__(self):
        """Return an iterator over the points.

        >>> list(PointSequence(Point(1.0, 2.0), Point(3.0, 4.0)))
        [Point(x=1.0, y=2.0), Point(x=3.0, y=4.0)]
        """
        return iter(self.points)

    def __reversed__(self):
        """Return a reversed iterator over the points.

        >>> list(reversed(PointSequence(Point(1.0, 2.0), Point(3.0, 4.0))))
        [Point(x=3.0, y=4.0), Point(x=1.0, y=2.0)]
        """
        return reversed(self.points)

    def __eq__(self, other: "PointSequence") -> bool:
        """Check if two polygons are equal.

        >>> ps1 = PointSequence(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0))
        >>> ps2 = PointSequence(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0))
        >>> ps1 == ps2
        True
        """
        return all(v1 == v2 for v1, v2 in zip(self.points, other.points))

    def __contains__(self, item: PointType | Iterable[Point]) -> bool:
        """Check if a point is in the sequence.

        >>> Point(1.0, 2.0) in PointSequence(Point(0.0, 0.0), Point(1.0, 2.0))
        True
        >>> (Point(1.0, 2.0), Point(3.0, 4.0)) in PointSequence(Point(0.0, 0.0), Point(1.0, 2.0))
        False
        >>> (Point(1.0, 2.0), Point(3.0, 4.0)) in PointSequence(Point(0.0, 0.0), Point(1.0, 2.0), Point(3.0, 4.0))
        True
        """
        if isinstance(item, Iterable) and isinstance(tuple(item)[0], Point):
            return all(p in self.points for p in item)
        return Point.aspoint(item) in self.points

    def append(self, point: PointType) -> None:
        """Add a point to the sequence.

        >>> ps = PointSequence(Point(1.0, 2.0), Point(3.0, 4.0))
        >>> ps.append(Point(5.0, 6.0))
        >>> ps
        PointSequence(Point(x=1.0, y=2.0), Point(x=3.0, y=4.0), Point(x=5.0, y=6.0))
        """
        self.points.append(Point.aspoint(point))

    def index(self, point: PointType, start: int = 0, stop: int = None) -> int:
        """Find the index of a point in the sequence.

        >>> ps = PointSequence(Point(1.0, 2.0), Point(3.0, 4.0))
        >>> ps.index(Point(3.0, 4.0))
        1

        Args:
            point: The point to find.
            start: The start index.
            stop: The end index.

        Returns:
            The index of the point.
        """
        args = (Point.aspoint(point), start, stop) if stop is not None else (Point.aspoint(point), start)
        return self.points.index(*args)
