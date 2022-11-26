from __future__ import annotations

from typing import Iterable

import numpy as np


class Geometry:
    #: Tolerance for equality.
    tolerance = 1e-6

    def __eq__(self, other: Geometry) -> bool:
        raise NotImplementedError

    def __ne__(self, other: Geometry) -> bool:
        return not self.__eq__(other)


class Point(Geometry):
    #: The x coordinate of the point.
    x: float
    #: The y coordinate of the point.
    y: float

    def __init__(self, *args: float | complex | Iterable[float]):
        """Initialize a point.

        >>> Point(1.0, 2.0)
        Point(x=1.0, y=2.0)
        >>> Point([1.0, 2.0])
        Point(x=1.0, y=2.0)
        >>> Point(complex(1.0, 2.0))
        Point(x=1.0, y=2.0)
        >>> Point(Point(1.0, 2.0))
        Point(x=1.0, y=2.0)
        >>> Point(1.0)
        Traceback (most recent call last):
        ...
        ValueError: Point must be initialized with two floats, a complex number or an iterable.

        Args:
            *args: two floats or a complex number or an iterable of two floats.
        """
        if len(args) == 1 and isinstance(args[0], complex):
            self.x, self.y = args[0].real, args[0].imag
        elif len(args) == 1 and isinstance(args[0], Iterable):
            self.x, self.y, *extra_args = tuple(args[0])
        elif len(args) == 2:
            self.x, self.y = args
        else:
            raise ValueError("Point must be initialized with two floats, a complex number or an iterable.")

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

    def __eq__(self, other: Point | Iterable[float] | complex) -> bool:
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

    def __add__(self, other: Point | Iterable[float] | complex) -> Point:
        """Add a point or vector to the point.

        >>> Point(1.0, 2.0) + Point(3.0, 4.0)
        Point(x=4.0, y=6.0)

        Args:
            other: The other point.
        """
        other = self.aspoint(other)
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point | Iterable[float] | complex) -> Point:
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

    @classmethod
    def aspoint(cls, p: Point | Iterable[float] | complex) -> Point:
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

    def distance(self, other: Point | Iterable[float] | complex) -> float:
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

    def direction(self, other: Point | Iterable[float] | complex) -> float:
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

    def vector(self, other: Point | Iterable[float] | complex) -> Vector:
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
    def __matmul__(self, other: Point | Iterable[float] | complex) -> float:
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

    def __abs__(self) -> float:
        """Calculate the magnitude of the vector.

        >>> abs(Vector(3.0, 4.0))
        5.0

        Returns:
            The magnitude of the vector.
        """
        return np.linalg.norm(self.array)

    def dot(self, other: Point | Iterable[float] | complex) -> float:
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
    def asvector(cls, v: Point | Vector | Iterable[float] | complex) -> Vector:
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


class LineCoefs:
    """The coefficients of a line in 2D space."""

    @classmethod
    def standardizeCoefficients(cls, A: float, B: float, C: float) -> tuple[float, float, float]:
        """Standardize the coefficients of a line.

        >>> Line.standardizeCoefficients(1.0, -1.0, 0.0)
        (1.0, -1.0, 0.0)
        >>> Line.standardizeCoefficients(-1.0, -1.0, 1.0)
        (1.0, 1.0, -1.0)

        Args:
            A: The coefficient of the x term.
            B: The coefficient of the y term.
            C: The constant term.

        Returns:
            The standardized coefficients.
        """
        if A != 0.0:
            A, B, C = 1.0, B / A, C / A
        elif B != 0.0:
            B, C = 1.0, C / B
        A, B, C = round(A + 0.0, 10), round(B + 0.0, 10), round(C + 0.0, 10)  # Prevent -0.0 and convert to float
        return A, B, C

    @classmethod
    def coefficientsFromTwoPoints(
        cls, p1: Point | Iterable[float] | complex, p2: Point | Iterable[float] | complex
    ) -> tuple[float, float, float]:
        """Get the coefficients of a line from two points.

        >>> LineCoefs.coefficientsFromTwoPoints(Point(0.0, 0.0), Point(1.0, 1.0))
        (1.0, -1.0, 0.0)

        Args:
            p1: The first point.
            p2: The second point.

        Returns:
            The coefficients of the line.
        """
        p1, p2 = Point.aspoint(p1), Point.aspoint(p2)
        A = p1.y - p2.y
        B = p2.x - p1.x
        C = p1.x * p2.y - p2.x * p1.y
        A, B, C = cls.standardizeCoefficients(A, B, C)
        return A, B, C

    @classmethod
    def coefficientsFromPointAndSlope(
        cls, p: Point | Iterable[float] | complex, slope: float
    ) -> tuple[float, float, float]:
        """Get the coefficients of a line from a point and a slope.

        >>> LineCoefs.coefficientsFromPointAndSlope(Point(0.0, 0.0), 1.0)
        (1.0, -1.0, 0.0)

        Args:
            p: The point.
            slope: The slope.

        Returns:
            The coefficients of the line.
        """
        p = Point.aspoint(p)
        A = slope
        B = -1.0
        C = p.y - slope * p.x
        A, B, C = cls.standardizeCoefficients(A, B, C)
        return A, B, C

    @classmethod
    def coefficientsFromSlopeAndIntercept(cls, slope: float, intercept: float) -> tuple[float, float, float]:
        """Get the coefficients of a line from a slope and an intercept.

        >>> LineCoefs.coefficientsFromSlopeAndIntercept(1.0, 0.0)
        (1.0, -1.0, 0.0)

        Args:
            slope: The slope.
            intercept: The intercept.

        Returns:
            The coefficients of the line.
        """
        A = slope
        B = -1.0
        C = intercept
        A, B, C = cls.standardizeCoefficients(A, B, C)
        return A, B, C

    @classmethod
    def coefficientsFromPointAndAngle(
        cls, p: Point | Iterable[float] | complex, angle: float
    ) -> tuple[float, float, float]:
        """Get the coefficients of a line from a point and an angle.

        >>> LineCoefs.coefficientsFromPointAndAngle(Point(0.0, 0.0), np.pi / 4.0)
        (1.0, -1.0, 0.0)

        Args:
            p: The point.
            angle: The angle.

        Returns:
            The coefficients of the line.
        """
        p = Point.aspoint(p)
        A = np.cos(angle)
        B = -np.sin(angle)
        C = B * p.y - A * p.x
        A, B, C = cls.standardizeCoefficients(A, B, C)
        return A, B, C

    @classmethod
    def coefficientsFromAngleAndIntercept(cls, angle: float, intercept: float) -> tuple[float, float, float]:
        """Get the coefficients of a line from an angle and an intercept.

        >>> LineCoefs.coefficientsFromAngleAndIntercept(np.pi / 4.0, 0.0)
        (1.0, -1.0, 0.0)

        Args:
            angle: The angle.
            intercept: The intercept.

        Returns:
            The coefficients of the line.
        """
        A = np.cos(angle)
        B = -np.sin(angle)
        C = intercept * np.sin(angle)
        A, B, C = cls.standardizeCoefficients(A, B, C)
        return A, B, C


class Line(Geometry, LineCoefs):
    #: Coefficient of the x term.
    A: float
    #: Coefficient of the y term.
    B: float
    #: Constant term.
    C: float

    def __init__(
        self,
        *,
        start: Point | Iterable[float] | complex = None,
        end: Point | Iterable[float] | complex = None,
        A: float = None,
        B: float = None,
        C: float = None,
        slope: float = None,
        angle: float = None,
        intercept: float = None,
    ):
        """Create a line. Possible ways to create a line (in order of precedence):

        - start and end points (start and end)
        - A, B, C coefficients (A, B and C)
        - start point and slope (start and slope)
        - start point and angle (start and angle)
        - slope and intercept (slope and intercept)
        - angle and intercept (angle and intercept)

        >>> Line(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        Line(A=1.0, B=-1.0, C=0.0)

        >>> Line(A=1.0, B=-1.0, C=0.0)
        Line(A=1.0, B=-1.0, C=0.0)

        >>> Line(start=Point(0.0, 0.0), slope=1.0)
        Line(A=1.0, B=-1.0, C=0.0)

        >>> Line(start=Point(0.0, 0.0), angle=np.pi / 4.0)
        Line(A=1.0, B=-1.0, C=0.0)

        >>> Line(slope=1.0, intercept=0.0)
        Line(A=1.0, B=-1.0, C=0.0)

        >>> Line(angle=np.pi / 4.0, intercept=0.0)
        Line(A=1.0, B=-1.0, C=0.0)

        >>> Line(start=Point(0.0, 0.0))
        Traceback (most recent call last):
        ...
        ValueError: Not enough information to create a line.

        Args:
            start: The start point of the line.
            end: The end point of the line.
            A: The coefficient of the x term.
            B: The coefficient of the y term.
            C: The constant term.
            slope: The slope of the line.
            intercept: The y-intercept of the line.
        """
        if start is not None and end is not None:
            A, B, C = self.coefficientsFromTwoPoints(start, end)
        elif A is not None and B is not None and C is not None:
            A, B, C = self.standardizeCoefficients(A, B, C)
        elif start is not None and slope is not None:
            A, B, C = self.coefficientsFromPointAndSlope(start, slope)
        elif start is not None and angle is not None:
            A, B, C = self.coefficientsFromPointAndAngle(start, angle)
        elif slope is not None and intercept is not None:
            A, B, C = self.coefficientsFromSlopeAndIntercept(slope, intercept)
        elif angle is not None and intercept is not None:
            A, B, C = self.coefficientsFromAngleAndIntercept(angle, intercept)
        else:
            raise ValueError("Not enough information to create a line.")
        self.A, self.B, self.C = A, B, C

    @classmethod
    def fromCoefficients(cls, A: float, B: float, C: float) -> Line:
        """Create a line from the coefficients.

        >>> Line.fromCoefficients(1.0, -1.0, 0.0)
        Line(A=1.0, B=-1.0, C=0.0)

        Args:
            A: The coefficient of the x term.
            B: The coefficient of the y term.
            C: The constant term.

        Returns:
            The created line.
        """
        A, B, C = cls.standardizeCoefficients(A, B, C)
        return cls(A=A, B=B, C=C)

    @classmethod
    def fromTwoPoints(cls, start: Point | Iterable[float] | complex, end: Point | Iterable[float] | complex) -> Line:
        """Create a line from two points.

        >>> Line.fromTwoPoints(Point(0.0, 0.0), Point(1.0, 1.0))
        Line(A=1.0, B=-1.0, C=0.0)

        Args:
            start: The first point.
            end: The second point.

        Returns:
            The created line.
        """
        start, end = Point.aspoint(start), Point.aspoint(end)
        A, B, C = cls.coefficientsFromTwoPoints(start, end)
        return cls.fromCoefficients(A=A, B=B, C=C)

    @classmethod
    def fromPointAndSlope(cls, start: Point | Iterable[float] | complex, slope: float) -> Line:
        """Create a line from a point and a slope.

        >>> Line.fromPointAndSlope(Point(0.0, 0.0), 1.0)
        Line(A=1.0, B=-1.0, C=0.0)

        Args:
            start: The point.
            slope: The slope.

        Returns:
            The created line.
        """
        start = Point.aspoint(start)
        A, B, C = cls.coefficientsFromPointAndSlope(start, slope)
        return cls.fromCoefficients(A=A, B=B, C=C)

    @classmethod
    def fromPointAndAngle(cls, start: Point | Iterable[float] | complex, angle: float) -> Line:
        """Create a line from a point and an angle.

        >>> Line.fromPointAndAngle(Point(0.0, 0.0), np.pi / 4.0)
        Line(A=1.0, B=-1.0, C=0.0)

        Args:
            start: The point.
            angle: The angle.

        Returns:
            The created line.
        """
        start = Point.aspoint(start)
        A, B, C = cls.coefficientsFromPointAndAngle(start, angle)
        return cls.fromCoefficients(A=A, B=B, C=C)

    @classmethod
    def fromSlopeAndIntercept(cls, slope: float, intercept: float) -> Line:
        """Create a line from a slope and intercept.

        >>> Line.fromSlopeAndIntercept(1.0, 0.0)
        Line(A=1.0, B=-1.0, C=0.0)

        Args:
            slope: The slope.
            intercept: The intercept.

        Returns:
            The created line.
        """
        A, B, C = cls.coefficientsFromSlopeAndIntercept(slope, intercept)
        return cls.fromCoefficients(A=A, B=B, C=C)

    @classmethod
    def fromAngleAndIntercept(cls, angle: float, intercept: float) -> Line:
        """Create a line from an angle and intercept.

        >>> Line.fromAngleAndIntercept(np.pi / 4.0, 0.0)
        Line(A=1.0, B=-1.0, C=0.0)

        Args:
            angle: The angle.
            intercept: The intercept.

        Returns:
            The created line.
        """
        A, B, C = cls.coefficientsFromAngleAndIntercept(angle, intercept)
        return cls.fromCoefficients(A=A, B=B, C=C)

    def __repr__(self) -> str:
        """Return the representation of the line.

        >>> Line(A=1.0, B=-1.0, C=0.0)
        Line(A=1.0, B=-1.0, C=0.0)
        """
        A, B, C = round(self.A, 10), round(self.B, 10), round(self.C, 10)
        return f"Line(A={A}, B={B}, C={C})"

    def __eq__(self, other: Line) -> bool:
        """Return whether the line is equal to another line.

        >>> Line(A=1.0, B=1.0, C=0.0) == Line(A=-1.0, B=-1.0, C=0.0)
        True
        >>> Line(A=1.0, B=1.0, C=0.0) == Line(A=1.0, B=2.0, C=1.0)
        False
        """
        multiplier = self.A / other.A if other.A != 0 else self.B / other.B if other.B != 0 else self.C / other.C
        return np.allclose(
            [self.A, self.B, self.C],
            [other.A * multiplier, other.B * multiplier, other.C * multiplier],
            atol=self.tolerance,
        )

    def __contains__(self, p: Point | Iterable[float] | complex) -> bool:
        """Check if a point is on this line.

        >>> Point(0.0, 0.0) in Line(A=-1.0, B=1.0, C=0.0)
        True
        >>> Point(0.0, 0.0) not in Line(A=-1.0, B=1.0, C=0.0)
        False
        >>> Point(1.0, 0.0) in Line(A=-1.0, B=1.0, C=0.0)
        False
        >>> Point(1.0, 0.0) not in Line(A=-1.0, B=1.0, C=0.0)
        True

        Returns:
            True if the point is on this line, otherwise False.
        """
        p = Point.aspoint(p)
        return self.distance(p) < self.tolerance

    def distance(self, p: Point) -> float:
        """Get the distance between a point and this line.

        >>> line = Line(A=1.0, B=1.0, C=0.0)
        >>> point = Point(1.0, 1.0)
        >>> assert np.isclose(line.distance(point), np.sqrt(2.0))

        Args:
            p: The point to get the distance to.

        Returns:
            The distance between the point and this line.
        """
        p = Point.aspoint(p)
        return abs(self.A * p.x + self.B * p.y + self.C) / np.sqrt(self.A**2 + self.B**2)

    @property
    def slope(self) -> float:
        """Get the slope of this line.

        >>> Line(A=1.0, B=-1.0, C=0.0).slope
        1.0
        >>> Line(A=1.0, B=1.0, C=0.0).slope
        -1.0
        >>> Line(A=1.0, B=0.0, C=0.0).slope
        inf

        Returns:
            The slope of this line.
        """
        if abs(self.B) < self.tolerance:
            return np.inf if self.A > 0 else -np.inf
        return -self.A / self.B

    @property
    def angle(self) -> float:
        """Get the angle of this line.

        >>> angle = Line(A=1.0, B=1.0, C=0.0).angle
        >>> assert np.isclose(angle, -np.pi / 4.0)

        Returns:
            The angle of this line.
        """
        return np.arctan2(-self.A, self.B)

    @property
    def intercept(self) -> float:
        """Get the intercept of this line.

        >>> Line(A=1.0, B=1.0, C=1.0).intercept
        -1.0
        >>> Line(A=1.0, B=0.0, C=0.0).intercept
        Traceback (most recent call last):
        ...
        ValueError: The line is vertical and has no intercept.

        Returns:
            The intercept of this line.
        """
        if abs(self.B) < self.tolerance:
            raise ValueError("The line is vertical and has no intercept.")
        return self.gety(0.0)

    def getx(self, y: float) -> float:
        """Get the x coordinate of a point on this line.

        >>> Line(A=1.0, B=1.0, C=0.0).getx(1.0)
        -1.0
        >>> Line(A=0.0, B=1.0, C=0.0).getx(0.0)
        Traceback (most recent call last):
        ...
        ValueError: Line is vertical

        Args:
            y: The y coordinate of the point.

        Returns:
            The x coordinate of the point.
        """
        if abs(self.A) < self.tolerance:
            raise ValueError("Line is vertical")
        return -(self.B * y + self.C) / self.A

    def gety(self, x: float) -> float:
        """Get the y coordinate of a point on this line.

        >>> Line(A=1.0, B=1.0, C=0.0).gety(1.0)
        -1.0
        >>> Line(A=1.0, B=0.0, C=0.0).gety(0.0)
        Traceback (most recent call last):
        ...
        ValueError: Line is horizontal

        Args:
            x: The x coordinate of the point.

        Returns:
            The y coordinate of the point.
        """
        if abs(self.B) < self.tolerance:
            raise ValueError("Line is horizontal")
        return -(self.A * x + self.C) / self.B

    def isParallel(self, line: "Line") -> bool:
        """Check if this line is parallel to another line.

        >>> Line(A=1.0, B=1.0, C=0.0).isParallel(Line(A=1.0, B=1.0, C=1.0))
        True
        >>> Line(A=1.0, B=1.0, C=0.0).isParallel(Line(A=-1.0, B=1.0, C=1.0))
        False

        Args:
            line: The line to check.

        Returns:
            True if the lines are parallel, otherwise False.
        """
        return abs(self.A * line.B - self.B * line.A) < self.tolerance

    def isPerpendicular(self, line: "Line") -> bool:
        """Check if this line is perpendicular to another line.

        >>> Line(A=1.0, B=1.0, C=0.0).isPerpendicular(Line(A=1.0, B=1.0, C=1.0))
        False
        >>> Line(A=1.0, B=1.0, C=0.0).isPerpendicular(Line(A=-1.0, B=1.0, C=1.0))
        True

        Args:
            line: The line to check.

        Returns:
            True if the lines are perpendicular, otherwise False.
        """
        return abs(self.A * line.A + self.B * line.B) < self.tolerance

    def intersect(self, line: "Line") -> Point:
        """Get the intersection point between this line and another line.

        >>> Line(A=1.0, B=1.0, C=-1.0).intersect(Line(A=1.0, B=-1.0, C=1.0))
        Point(x=0.0, y=1.0)
        >>> Line(A=1.0, B=1.0, C=0.0).intersect(Line(A=1.0, B=1.0, C=1.0))
        Traceback (most recent call last):
        ...
        ValueError: Lines are parallel and do not intersect

        Args:
            line: The line to intersect with.

        Returns:
            The intersection point if there is one, otherwise None.
        """
        if self.isParallel(line):
            raise ValueError("Lines are parallel and do not intersect")
        A = np.asarray([[self.A, self.B], [line.A, line.B]])
        b = np.asarray([-self.C, -line.C])
        x, y = np.linalg.solve(A, b)
        p = Point(x, y)
        assert p in self and p in line, "Intersection point is not on both lines"
        return p

    def parallel(self, p: Point | Iterable[float] | complex) -> "Line":
        """Get a parallel line to this line.

        >>> Line(A=1.0, B=-1.0, C=0.0).parallel(Point(0.0, 1.0))
        Line(A=1.0, B=-1.0, C=1.0)

        Args:
            p: A point on the parallel line.

        Returns:
            A parallel line to this line.
        """
        p = Point.aspoint(p)
        return Line(A=self.A, B=self.B, C=-self.A * p.x - self.B * p.y)

    def perpendicular(self, p: Point | Iterable[float] | complex) -> "Line":
        """Get a perpendicular line to this line.

        >>> Line(A=1.0, B=-1.0, C=0.0).perpendicular(Point(0.0, 1.0))
        Line(A=1.0, B=1.0, C=-1.0)
        >>> Line(A=0.0, B=1.0, C=0.0).perpendicular(Point(0.0, 0.0))
        Line(A=1.0, B=0.0, C=0.0)
        >>> Line(A=1.0, B=0.0, C=0.0).perpendicular(Point(0.0, 0.0))
        Line(A=0.0, B=1.0, C=0.0)

        Args:
            p: A point on the perpendicular line.

        Returns:
            A perpendicular line to this line.
        """
        p = Point.aspoint(p)
        return Line(A=self.B, B=-self.A, C=-self.B * p.x + self.A * p.y)


class Segment(Line):
    #: The first point to create the line.
    start: Point
    #: The second point to create the line.
    end: Point

    def __init__(self, *, start: Point | Iterable[float] | complex, end: Point | Iterable[float] | complex):
        """Create a new line segment.

        >>> Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        Segment(start=Point(x=0.0, y=0.0), end=Point(x=1.0, y=1.0)) -> Line(A=1.0, B=-1.0, C=0.0)

        Args:
            start: The first point to create the line.
            end: The second point to create the line.
        """
        super().__init__(start=start, end=end)
        self.start, self.end = Point.aspoint(start), Point.aspoint(end)

    def __repr__(self):
        """Get the string representation of this line segment.

        >>> repr(Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0)))
        'Segment(start=Point(x=0.0, y=0.0), end=Point(x=1.0, y=1.0)) -> Line(A=1.0, B=-1.0, C=0.0)'
        """
        return f"Segment(start={self.start}, end={self.end}) -> {super().__repr__()}"

    def __eq__(self, other: Segment) -> bool:
        """Check if this line segment is equal to another line segment.

        >>> Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0)) == Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        True
        >>> Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0)) == Segment(start=Point(1.0, 1.0), end=Point(0.0, 0.0))
        True
        >>> Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0)) == Segment(start=Point(0.0, 0.0), end=Point(2.0, 2.0))
        False

        Args:
            other: The line segment to check.
        """
        return super().__eq__(other) and (
            (self.start == other.start and self.end == other.end)
            or (self.start == other.end and self.end == other.start)
        )

    def __contains__(self, p: Point | Iterable[float] | complex) -> bool:
        """Check if a point is on this segment.

        >>> Point(0.0, 0.0) in Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        True
        >>> Point(1.0, 1.0) in Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        True
        >>> Point(0.5, 0.5) in Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        True
        >>> Point(0.0, 1.0) in Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        False

        Returns:
            True if the point is on this segment, otherwise False.
        """
        p = Point.aspoint(p)
        if not super().__contains__(p):
            return False
        minx, maxx = sorted([self.start.x, self.end.x])
        miny, maxy = sorted([self.start.y, self.end.y])
        return minx <= p.x <= maxx and miny <= p.y <= maxy

    @property
    def length(self) -> float:
        """Get the length of this segment.

        >>> Segment(start=Point(0.0, 0.0), end=Point(3.0, 4.0)).length
        5.0

        Returns:
            The length of this segment.
        """
        return self.start.distance(self.end)

    @property
    def direction(self) -> float:
        """Get the direction of this segment.

        >>> Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0)).direction - np.pi / 4
        0.0
        >>> Segment(start=Point(0.0, 0.0), end=Point(-1.0, 1.0)).direction - 3.0 * np.pi / 4
        0.0
        >>> Segment(start=Point(0.0, 0.0), end=Point(-1.0, -1.0)).direction + 3.0 * np.pi / 4.0
        0.0
        >>> Segment(start=Point(0.0, 0.0), end=Point(1.0, -1.0)).direction + np.pi / 4.0
        0.0

        Returns:
            The direction of this segment.
        """
        return self.start.direction(self.end)

    @property
    def midpoint(self) -> Point:
        """Get the midpoint of this segment.

        >>> Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0)).midpoint
        Point(x=0.5, y=0.5)

        Returns:
            The midpoint of this segment.
        """
        return Point((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)

    def reverse(self) -> "Segment":
        """Reverse the direction of this segment.

        >>> Segment(start=Point(0.0, 0.0), end=Point(1.0, 1.0)).reverse()
        Segment(start=Point(x=1.0, y=1.0), end=Point(x=0.0, y=0.0)) -> Line(A=1.0, B=-1.0, C=0.0)
        """
        self.start, self.end = self.end, self.start
        return self


class Ray(Line):
    #: The first point to create the line.
    start: Point
    #: The second point to create the line.
    end: Point

    def __init__(
        self,
        *,
        start: Point | Iterable[float] | complex,
        end: Point | Iterable[float] | complex,
    ):
        """Create a ray.

        >>> Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        Ray(start=Point(x=0.0, y=0.0), slope=1.0) -> Line(A=1.0, B=-1.0, C=0.0)

        Args:
            start: The first point to create the line.
            end: The second point to create the line.
        """
        super().__init__(start=start, end=end)
        self.start, self.end = Point.aspoint(start), Point.aspoint(end)
        self.direction = self.end - self.start

    def __repr__(self):
        """Get the string representation of this ray.

        >>> Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        Ray(start=Point(x=0.0, y=0.0), slope=1.0) -> Line(A=1.0, B=-1.0, C=0.0)
        """
        slope = round(self.slope, 10)
        return f"Ray(start={self.start}, slope={slope}) -> {super().__repr__()}"

    def __eq__(self, other: Ray) -> bool:
        """Check if two rays are equal.

        >>> Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0)) == Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        True
        >>> Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0)) == Ray(start=Point(0.0, 0.0), end=Point(2.0, 2.0))
        True
        >>> Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0)) == Ray(start=Point(-1.0, -1.0), end=Point(1.0, 1.0))
        False
        """
        return super().__eq__(other) and self.start == other.start

    def __contains__(self, p: Point | Iterable[float] | complex):
        """Check if a point is on this ray.

        >>> Point(0.0, 0.0) in Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        True
        >>> Point(0.5, 0.5) in Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        True
        >>> Point(1.0, 1.0) in Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        True
        >>> Point(2.0, 2.0) in Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        True
        >>> Point(-1.0, -1.0) in Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0))
        False

        Returns:
            True if the point is on this ray, otherwise False.
        """
        return super().__contains__(p) and self.start.vector(p) @ self.slope_vector >= 0

    @property
    def slope_vector(self) -> Vector:
        """Get the slope vector of this ray.

        >>> Ray(start=Point(0.0, 0.0), end=Point(1.0, 1.0)).slope_vector
        Vector(x=1.0, y=1.0)

        Returns:
            The slope vector of this ray.
        """
        return Vector.asvector(self.end - self.start)


def test():
    import doctest

    doctest.testmod()
