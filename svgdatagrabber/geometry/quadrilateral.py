from .point import PointType
from .polygon import Polygon


class Quadrilateral(Polygon):
    def __init__(self, *points: PointType):
        if len(points) != 4:
            raise ValueError("A quadrilateral must have exactly four points.")
        super().__init__(*points)

    def check(self):
        """Check if the quadrilateral is convex and complex."""
        pass


class Kite(Quadrilateral):
    pass


class Rhombus(Kite):
    pass


class Trapezoid(Quadrilateral):
    pass


class IsoscelesTrapezoid(Trapezoid):
    pass


class Parallelogram(Quadrilateral):
    pass


class Rectangle(Parallelogram, IsoscelesTrapezoid):
    pass


class Square(Rectangle, Rhombus, Parallelogram):
    pass
