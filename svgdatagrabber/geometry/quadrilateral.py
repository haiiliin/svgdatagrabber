import numpy as np

from .line import Segment
from .point import Point, PointType
from .polygon import Polygon


class Quadrilateral(Polygon):
    def __init__(self, *points: PointType):
        if len(points) != 4:
            raise ValueError("A quadrilateral must have exactly four points.")
        super().__init__(*points)

    def check(self):
        """Check if the quadrilateral is convex and complex.

        >>> Quadrilateral(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)).check()
        >>> Quadrilateral(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, -1.0)).check()
        Traceback (most recent call last):
        ...
        AssertionError: The quadrilateral must be simple.
        """
        assert self.isSimple, "The quadrilateral must be simple."
        assert self.isValid, "The quadrilateral must be convex."


class Kite(Quadrilateral):
    @property
    def isKite(self):
        """Check if the quadrilateral is a kite."""
        (l1, l2, l3, l4) = (edge.length for edge in self.edges)  # type: float
        return np.allclose([l1, l3], [l2, l4]) or np.allclose([l2, l1], [l3, l4])

    def check(self):
        """Check if the quadrilateral is a kite.

        >>> Kite(Point(1.0, 0.0), Point(0.0, 1.0), Point(-1.0, 0.0), Point(0.0, -2.0)).check()
        >>> Kite(Point(2.0, 0.0), Point(0.0, 1.0), Point(-1.0, 0.0), Point(0.0, -2.0)).check()
        Traceback (most recent call last):
        ...
        AssertionError: Two pairs of adjacent sides must be of equal length.
        """
        super().check()
        assert self.isKite, "Two pairs of adjacent sides must be of equal length."


class RightKite(Kite):
    @property
    def isRightKite(self):
        """Check if the quadrilateral is a right kite."""
        e1, e2, e3, e4 = self.edges  # type: Segment
        l1, l2, l3, l4 = e1.length, e2.length, e3.length, e4.length
        return (np.allclose([l1, l3], [l2, l4]) and e1.isPerpendicular(e4)) or (
            np.allclose([l2, l1], [l3, l4]) and e1.isPerpendicular(e2)
        )

    def check(self):
        """Check if the quadrilateral is a right kite.

        >>> RightKite(Point(1.0, 0.0), Point(0.0, 1.0), Point(-1.0, 0.0), Point(0.0, -2.0)).check()
        Traceback (most recent call last):
        ...
        AssertionError: A kite must be inscribed in a circle.
        >>> RightKite(Point(3.0, 0.0), Point(0.0, 4.0), Point(-3.0, 0.0), Point(0.0, -2.25)).check()
        """
        super().check()
        assert self.isRightKite, "A kite must be inscribed in a circle."


class Rhombus(Kite):
    @property
    def isRhombus(self):
        """Check if the quadrilateral is a rhombus."""
        (l1, l2, l3, l4) = (edge.length for edge in self.edges)  # type: float
        return np.allclose([l2, l3, l4], l1)

    def check(self):
        """Check if the quadrilateral is a rhombus.

        >>> Rhombus(Point(1.0, 0.0), Point(0.0, 2.0), Point(-1.0, 0.0), Point(0.0, -2.0)).check()
        >>> Rhombus(Point(2.0, 0.0), Point(0.0, 2.0), Point(-1.0, 0.0), Point(0.0, -2.0)).check()
        Traceback (most recent call last):
        ...
        AssertionError: All sides must be of equal length.
        """
        super().check()
        assert self.isRhombus, "All sides must be of equal length."


class Trapezoid(Quadrilateral):
    @property
    def isTrapezoid(self):
        """Check if the quadrilateral is a trapezoid."""
        e1, e2, e3, e4 = self.edges  # type: Segment
        return e1.isParallel(e3) or e2.isParallel(e4)

    def check(self):
        """Check if the quadrilateral is a trapezoid.

        >>> Trapezoid(Point(2.0, 0.0), Point(1.0, 1.0), Point(-1.0, 1.0), Point(-3.0, 0.0)).check()
        >>> Trapezoid(Point(2.0, 0.0), Point(1.0, 1.0), Point(-1.0, 2.0), Point(-3.0, 0.0)).check()
        Traceback (most recent call last):
        ...
        AssertionError: Two pairs of adjacent sides must be parallel.
        """
        super().check()
        assert self.isTrapezoid, "Two pairs of adjacent sides must be parallel."


class IsoscelesTrapezoid(Trapezoid):
    @property
    def isIsoscelesTrapezoid(self):
        """Check if the quadrilateral is an isosceles trapezoid."""
        #: https://en.wikipedia.org/wiki/Isosceles_trapezoid
        e1, e2, e3, e4 = self.edges  # type: Segment
        l1, l2, l3, l4 = e1.length, e2.length, e3.length, e4.length
        return (e1.isParallel(e3) and np.allclose(l2, l4)) or (e2.isParallel(e4) and np.allclose(l1, l3))

    def check(self):
        """Check if the quadrilateral is an isosceles trapezoid.

        >>> IsoscelesTrapezoid(Point(2.0, 0.0), Point(1.0, 1.0), Point(-1.0, 1.0), Point(-2.0, 0.0)).check()
        >>> IsoscelesTrapezoid(Point(2.0, 0.0), Point(1.0, 1.0), Point(-1.0, 1.0), Point(-3.0, 0.0)).check()
        Traceback (most recent call last):
        ...
        AssertionError: Two pairs of adjacent sides must be parallel and of equal length.
        """
        super().check()
        assert self.isIsoscelesTrapezoid, "Two pairs of adjacent sides must be parallel and of equal length."


class Parallelogram(Quadrilateral):
    @property
    def isParallelogram(self):
        """Check if the quadrilateral is a parallelogram."""
        e1, e2, e3, e4 = self.edges  # type: Segment
        return e1.isParallel(e3) and e2.isParallel(e4)

    def check(self):
        """Check if the quadrilateral is a parallelogram.

        >>> Parallelogram(Point(0.0, 0.0), Point(2.0, 0.0), Point(3.0, 1.0), Point(1.0, 1.0)).check()
        >>> Parallelogram(Point(-1.0, 0.0), Point(2.0, 0.0), Point(3.0, 1.0), Point(1.0, 1.0)).check()
        Traceback (most recent call last):
        ...
        AssertionError: Two pairs of adjacent sides must be parallel.
        """
        super().check()
        assert self.isParallelogram, "Two pairs of adjacent sides must be parallel."


class Rectangle(Parallelogram, IsoscelesTrapezoid):
    @property
    def isRectangle(self):
        """Check if the quadrilateral is a rectangle."""
        e1, e2, e3, e4 = self.edges  # type: Segment
        return e1.isPerpendicular(e2)

    def check(self):
        """Check if the quadrilateral is a rectangle.

        >>> Rectangle(Point(0.0, 0.0), Point(2.0, 0.0), Point(2.0, 1.0), Point(0.0, 1.0)).check()
        >>> Rectangle(Point(0.0, 0.0), Point(2.0, 0.0), Point(3.0, 1.0), Point(1.0, 1.0)).check()
        Traceback (most recent call last):
        ...
        AssertionError: The quadrilateral must be a right parallelogram.
        """
        super().check()
        assert self.isRectangle, "The quadrilateral must be a right parallelogram."


class Square(Rectangle, Rhombus, Parallelogram):
    @property
    def isSquare(self):
        """Check if the quadrilateral is a square."""
        #: https://en.wikipedia.org/wiki/Square
        (l1, l2, l3, l4) = (edge.length for edge in self.edges)  # type: float
        return np.allclose([l1, l2, l3, l4], l1)

    def check(self):
        """Check if the quadrilateral is a square.

        >>> Square(Point(0.0, 0.0), Point(2.0, 0.0), Point(2.0, 2.0), Point(0.0, 2.0)).check()
        >>> Square(Point(0.0, 0.0), Point(2.0, 0.0), Point(2.0, 1.0), Point(0.0, 1.0)).check()
        Traceback (most recent call last):
        ...
        AssertionError: Two pairs of adjacent sides must be of equal length.
        """
        super().check()
        assert self.isSquare, "All sides must be of equal length."