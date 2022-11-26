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

    def __contains__(self, item: Geometry) -> bool:
        pass


class Point(Geometry):
    #: The x coordinate of the point.
    x: float
    #: The y coordinate of the point.
    y: float

    def __init__(self, *args: float):
        self.x, self.y, *extra_args = args

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other: Point) -> bool:
        return np.allclose([self.x, self.y], [other.x, other.y], atol=self.tolerance)

    @classmethod
    def aspoint(cls, p: Point | Iterable[float] | complex) -> Point:
        """Convert a point to a Point object.

        Args:
            p: The point to convert.

        Returns:
            The converted point.
        """
        if isinstance(p, Point):
            return p
        elif isinstance(p, complex):
            return cls(p.real, p.imag)
        return cls(*p)

    def distance(self, other: Point) -> float:
        """Calculate the distance between two points.

        Args:
            other: The other point.

        Returns:
            The distance between the points.
        """
        return np.linalg.norm(np.array([self.x, self.y]) - np.array([other.x, other.y]))

    def direction(self, other: Point) -> float:
        """Calculate the direction between two points.

        Args:
            other: The other point.

        Returns:
            The direction between the points.
        """
        return np.arctan2(other.y - self.y, other.x - self.x)


class Line2DCoefficients:
    """The coefficients of a line in 2D space."""

    @classmethod
    def coefficientsFromTwoPoints(
        cls, p1: Point | Iterable[float] | complex, p2: Point | Iterable[float] | complex
    ) -> tuple[float, float, float]:
        """Get the coefficients of a line from two points.

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
        return A, B, C

    @classmethod
    def coefficientsFromPointAndSlope(
        cls, p: Point | Iterable[float] | complex, slope: float
    ) -> tuple[float, float, float]:
        """Get the coefficients of a line from a point and a slope.

        Args:
            p: The point.
            slope: The slope.

        Returns:
            The coefficients of the line.
        """
        p = Point.aspoint(p)
        A = slope
        B = -1
        C = p.y - slope * p.x
        return A, B, C

    @classmethod
    def coefficientsFromSlopeAndIntercept(cls, slope: float, intercept: float) -> tuple[float, float, float]:
        """Get the coefficients of a line from a slope and an intercept.

        Args:
            slope: The slope.
            intercept: The intercept.

        Returns:
            The coefficients of the line.
        """
        A = slope
        B = -1
        C = intercept
        return A, B, C

    @classmethod
    def coefficientsFromPointAndAngle(
        cls, p: Point | Iterable[float] | complex, angle: float
    ) -> tuple[float, float, float]:
        """Get the coefficients of a line from a point and an angle.

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
        return A, B, C

    @classmethod
    def coefficientsFromAngleAndIntercept(cls, angle: float, intercept: float) -> tuple[float, float, float]:
        """Get the coefficients of a line from an angle and an intercept.

        Args:
            angle: The angle.
            intercept: The intercept.

        Returns:
            The coefficients of the line.
        """
        A = np.cos(angle)
        B = -np.sin(angle)
        C = intercept * np.sin(angle)
        return A, B, C


class Line(Geometry, Line2DCoefficients):
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
            pass
        elif start is not None and slope is not None:
            A, B, C = self.coefficientsFromPointAndSlope(start, slope)
        elif start is not None and angle is not None:
            A, B, C = self.coefficientsFromPointAndAngle(start, angle)
        elif slope is not None and intercept is not None:
            A, B, C = self.coefficientsFromSlopeAndIntercept(slope, intercept)
        elif angle is not None and intercept is not None:
            A, B, C = self.coefficientsFromAngleAndIntercept(angle, intercept)
        else:
            raise ValueError("Invalid arguments.")
        self.A, self.B, self.C = A, B, C

    @classmethod
    def fromCoefficients(cls, A: float, B: float, C: float) -> Line:
        """Create a line from the coefficients.

        Args:
            A: The coefficient of the x term.
            B: The coefficient of the y term.
            C: The constant term.

        Returns:
            The created line.
        """
        obj = cls()
        obj.A, obj.B, obj.C = A, B, C
        return obj

    @classmethod
    def fromTwoPoints(cls, start: Point | Iterable[float] | complex, end: Point | Iterable[float] | complex) -> Line:
        """Create a line from two points.

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

        Args:
            angle: The angle.
            intercept: The intercept.

        Returns:
            The created line.
        """
        A, B, C = cls.coefficientsFromAngleAndIntercept(angle, intercept)
        return cls.fromCoefficients(A=A, B=B, C=C)

    def __repr__(self):
        return f"Line ({self.A} * x {self.B:+} * y {self.C:+} = 0)"

    def __eq__(self, other: Line) -> bool:
        return np.allclose([self.A, self.B, self.C], [other.A, other.B, other.C], atol=self.tolerance)

    def __contains__(self, p: Point | Iterable[float] | complex) -> bool:
        """Check if a point is on this line.

        Returns:
            True if the point is on this line, otherwise False.
        """
        p = Point.aspoint(p)
        return self.distance(p) < self.tolerance

    def distance(self, p: Point) -> float:
        """Get the distance between a point and this line.

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

        Returns:
            The slope of this line.
        """
        if abs(self.B) < self.tolerance:
            return np.inf
        return self.A / self.B

    @property
    def intercept(self) -> float:
        """Get the intercept of this line.

        Returns:
            The intercept of this line.
        """
        if abs(self.B) < self.tolerance:
            return np.inf
        return -self.C / self.B

    def getx(self, y: float) -> float:
        """Get the x coordinate of a point on this line.

        Args:
            y: The y coordinate of the point.

        Returns:
            The x coordinate of the point.
        """
        if abs(self.A) < self.tolerance:
            raise ValueError("Line is vertical")
        return (self.C - self.B * y) / self.A

    def gety(self, x: float) -> float:
        """Get the y coordinate of a point on this line.

        Args:
            x: The x coordinate of the point.

        Returns:
            The y coordinate of the point.
        """
        if abs(self.B) < self.tolerance:
            raise ValueError("Line is horizontal")
        return (self.C - self.A * x) / self.B

    def isParallel(self, line: "Line") -> bool:
        """Check if this line is parallel to another line.

        Args:
            line: The line to check.

        Returns:
            True if the lines are parallel, otherwise False.
        """
        return abs(self.A * line.B - self.B * line.A) < self.tolerance

    def isPerpendicular(self, line: "Line") -> bool:
        """Check if this line is perpendicular to another line.

        Args:
            line: The line to check.

        Returns:
            True if the lines are perpendicular, otherwise False.
        """
        return abs(self.A * line.A + self.B * line.B) < self.tolerance

    def intersect(self, line: "Line") -> Point:
        """Get the intersection point between this line and another line.

        Args:
            line: The line to intersect with.

        Returns:
            The intersection point if there is one, otherwise None.
        """
        if self.isParallel(line):
            raise ValueError("Lines are parallel and do not intersect")
        A = np.asarray([[self.A, self.B], [line.A, line.B]])
        b = np.asarray([self.C, line.C])
        try:
            x, y = np.linalg.solve(A, b)
            p = Point(x, y)
            assert p in self and p in line
            return p
        except np.linalg.LinAlgError:
            raise ValueError("Lines are parallel and do not intersect")
        except AssertionError:
            raise ValueError("Intersection point is not on both lines")

    def parallel(self, p: Point | Iterable[float] | complex) -> "Line":
        """Get a parallel line to this line.

        Args:
            p: A point on the parallel line.

        Returns:
            A parallel line to this line.
        """
        p = Point.aspoint(p)
        return Line(A=self.A, B=self.B, C=-self.A * p.x - self.B * p.y)

    def perpendicular(self, p: Point | Iterable[float] | complex) -> "Line":
        """Get a perpendicular line to this line.

        Args:
            p: A point on the perpendicular line.

        Returns:
            A perpendicular line to this line.
        """
        p = Point.aspoint(p)
        return Line(start=p, end=Point(p.x - self.B, p.y + self.A))


class Segment(Line):
    #: The first point to create the line.
    start: Point
    #: The second point to create the line.
    end: Point

    def __init__(self, *, start: Point | Iterable[float] | complex, end: Point | Iterable[float] | complex):
        super().__init__(start=start, end=end)
        self.start, self.end = Point.aspoint(start), Point.aspoint(end)

    def __repr__(self):
        return f"Segment ({self.start}, {self.end}) -> {super().__repr__()})"

    def __eq__(self, other: Segment) -> bool:
        return super().__eq__(other) and self.start == other.start and self.end == other.end

    def __contains__(self, p: Point | Iterable[float] | complex) -> bool:
        """Check if a point is on this segment.

        Returns:
            True if the point is on this segment, otherwise False.
        """
        p = Point.aspoint(p)
        if not super().__contains__(p):
            return False
        minx, maxx = sorted([self.start.x, self.end.x])
        miny, maxy = sorted([self.start.y, self.end.y])
        return minx <= p.x <= maxx and miny <= p.y <= maxy

    def __gt__(self, other: Segment) -> bool:
        return self.length > other.length

    def __ge__(self, other: Segment) -> bool:
        return self.length >= other.length

    def __lt__(self, other: Segment) -> bool:
        return self.length < other.length

    def __le__(self, other: Segment) -> bool:
        return self.length <= other.length

    @property
    def length(self) -> float:
        """Get the length of this segment.

        Returns:
            The length of this segment.
        """
        return self.start.distance(self.end)

    @property
    def direction(self) -> float:
        """Get the direction of this segment.

        Returns:
            The direction of this segment.
        """
        return self.start.direction(self.end)

    def midpoint(self) -> Point:
        """Get the midpoint of this segment.

        Returns:
            The midpoint of this segment.
        """
        return Point((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)

    def reverse(self):
        """Reverse the direction of this segment."""
        self.start, self.end = self.end, self.start


class Ray(Line):
    #: The first point to create the line.
    origin: Point

    def __init__(
        self,
        *,
        origin: Point | Iterable[float] | complex,
        end: Point | Iterable[float] | complex = None,
        slope: float = None,
        angle: float = None,
    ):
        """Create a ray. Possible ways to create a ray (in order of precedence):

        - origin and end points (end point is not included in the ray)
        - origin and slope (slope is not included in the ray)
        - origin and angle (angle is not included in the ray)
        """
        super().__init__(start=origin, end=end, slope=slope, angle=angle)
        self.origin = Point.aspoint(origin)

    def __repr__(self):
        return f"Ray ({self.origin}, slope={self.slope}) -> {super().__repr__()})"

    def __eq__(self, other: Ray) -> bool:
        return super().__eq__(other) and self.origin == other.origin

    def __contains__(self, p: Point | Iterable[float] | complex):
        """Check if a point is on this ray.

        Returns:
            True if the point is on this ray, otherwise False.
        """
        p = Point.aspoint(p)
        if not super().__contains__(p):
            return False
        return (
            (self.origin.x <= p.x and self.origin.y <= p.y)
            if self.slope > 0
            else (self.origin.x >= p.x and self.origin.y >= p.y)
        )