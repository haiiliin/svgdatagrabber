from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Iterable, Sequence, overload, List

from .closedshape import ClosedShape
from .geometrybase import GeometryBase
from .point import Point, PointType


class IterablePoint(GeometryBase, ABC):
    def __iter__(self) -> Iterable[Point]:
        """Iterate over the geometry vertices."""
        raise NotImplementedError


class StraightLineShape(IterablePoint, ABC):
    def within(self, other: "ClosedShape") -> bool:
        return all(p in other for p in self)


class PointSequence(Sequence[Point], IterablePoint):
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
