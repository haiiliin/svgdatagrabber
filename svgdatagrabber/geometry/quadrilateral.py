import numpy as np

from .line import Segment
from .point import PointType
from .polygon import Polygon


class Quadrilateral(Polygon):
    def __init__(self, *points: PointType):
        if len(points) != 4:
            raise ValueError("A quadrilateral must have exactly four points.")
        super().__init__(*points)

    def check(self):
        """Check if the quadrilateral is convex and complex."""
        assert self.isSimple(), "The quadrilateral must be simple."
        assert self.isConvex(), "The quadrilateral must be convex."


class Kite(Quadrilateral):
    def isKite(self):
        """Check if the quadrilateral is a kite."""
        #: https://en.wikipedia.org/wiki/Kite_(geometry)
        (l1, l2, l3, l4) = (edge.length for edge in self.edges)  # type: float
        return np.allclose([l1, l2], [l3, l4]) or np.allclose([l2, l3], [l1, l4])

    def check(self):
        """Check if the quadrilateral is a kite."""
        super().check()
        assert self.isKite(), "Two pairs of adjacent sides must be of equal length"


class RightKite(Kite):
    def isRightKite(self):
        """Check if the quadrilateral is a right kite."""
        #: https://en.wikipedia.org/wiki/Kite_(geometry)
        e1, e2, e3, e4 = self.edges  # type: Segment
        l1, l2, l3, l4 = e1.length, e2.length, e3.length, e4.length
        return (np.allclose([l1, l2], [l3, l4]) and e2.isPerpendicular(e3)) or (
            np.allclose([l2, l3], [l1, l4]) and e1.isPerpendicular(e2)
        )

    def check(self):
        """Check if the quadrilateral is a right kite."""
        super().check()
        assert self.isRightKite(), "A kite must be inscribed in a circle."


class Rhombus(Kite):
    def isRhombus(self):
        """Check if the quadrilateral is a rhombus."""
        #: https://en.wikipedia.org/wiki/Rhombus
        (l1, l2, l3, l4) = (edge.length for edge in self.edges)  # type: float
        return np.allclose([l2, l3, l4], l1)

    def check(self):
        """Check if the quadrilateral is a rhombus."""
        super().check()
        assert self.isRhombus(), "All sides must be of equal length."


class Trapezoid(Quadrilateral):
    def isTrapezoid(self):
        """Check if the quadrilateral is a trapezoid."""
        #: https://en.wikipedia.org/wiki/Trapezoid
        e1, e2, e3, e4 = self.edges  # type: Segment
        return e1.isParallel(e3) or e2.isParallel(e4)

    def check(self):
        """Check if the quadrilateral is a trapezoid."""
        super().check()
        assert self.isTrapezoid(), "Two pairs of adjacent sides must be parallel."


class IsoscelesTrapezoid(Trapezoid):
    def isIsoscelesTrapezoid(self):
        """Check if the quadrilateral is an isosceles trapezoid."""
        #: https://en.wikipedia.org/wiki/Isosceles_trapezoid
        e1, e2, e3, e4 = self.edges  # type: Segment
        l1, l2, l3, l4 = e1.length, e2.length, e3.length, e4.length
        return (e1.isParallel(e3) and np.allclose(l2, l4) and not e2.isParallel(e4)) or (
            e2.isParallel(e4) and np.allclose(l1, l3) and not e1.isParallel(e3)
        )

    def check(self):
        """Check if the quadrilateral is an isosceles trapezoid."""
        super().check()
        assert self.isIsoscelesTrapezoid(), "Two pairs of adjacent sides must be parallel and of equal length."


class Parallelogram(Quadrilateral):
    def isParallelogram(self):
        """Check if the quadrilateral is a parallelogram."""
        #: https://en.wikipedia.org/wiki/Parallelogram
        e1, e2, e3, e4 = self.edges  # type: Segment
        return e1.isParallel(e3) and e2.isParallel(e4)

    def check(self):
        """Check if the quadrilateral is a parallelogram."""
        super().check()
        assert self.isParallelogram(), "Two pairs of adjacent sides must be parallel."


class Rectangle(Parallelogram, IsoscelesTrapezoid):
    def isRectangle(self):
        """Check if the quadrilateral is a rectangle."""
        #: https://en.wikipedia.org/wiki/Rectangle
        e1, e2, e3, e4 = self.edges  # type: Segment
        return e1.isPerpendicular(e2) and e3.isPerpendicular(e4)

    def check(self):
        """Check if the quadrilateral is a rectangle."""
        super().check()
        assert self.isRectangle(), "The quadrilateral must be a right parallelogram."


class Square(Rectangle, Rhombus, Parallelogram):
    def isSquare(self):
        """Check if the quadrilateral is a square."""
        #: https://en.wikipedia.org/wiki/Square
        (l1, l2, l3, l4) = (edge.length for edge in self.edges)  # type: float
        return np.allclose([l1, l2, l3, l4], l1)

    def check(self):
        """Check if the quadrilateral is a square."""
        super().check()
        assert self.isSquare(), "All sides must be of equal length."
