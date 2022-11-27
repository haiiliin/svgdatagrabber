from __future__ import annotations

from typing import List, Iterable, Tuple

from .closedshape import ClosedShape
from .line import Ray, Segment
from .point import Point, PointType
from .pointsequence import PointSequence, IterablePoint


class Polygon(PointSequence, ClosedShape, IterablePoint):
    def __init__(self, *points: PointType):
        """Create a polygon.

        >>> Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0))
        Polygon(Point(x=0.0, y=0.0), Point(x=1.0, y=0.0), Point(x=1.0, y=1.0), Point(x=0.0, y=1.0))

        Args:
            points: The vertices of the polygon.
        """
        if len(points) < 3:
            raise ValueError("A polygon must have at least three points.")
        elif not self.isConvex():
            raise ValueError("A polygon must be convex.")
        super().__init__(*points)

    def __contains__(self, item: PointType | Iterable[Point]) -> bool:
        """Check if a point or shape is inside the shape.

        >>> polygon = Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0))
        >>> (Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)) in polygon
        True
        >>> Point(0.5, 0.5) in polygon
        True
        >>> (Point(0.5, 0.0), Point(0.5, 1.0), Point(0.0, 0.5), Point(1.0, 0.5)) in polygon
        True
        >>> Point(0.5, 1.5) in polygon
        False

        Args:
            item: The point or shape.
        """
        return self.contains(item)

    def isConvex(self):
        """Check if the polygon is convex."""
        #: https://stackoverflow.com/questions/471962/how-do-determine-if-a-polygon-is-complex-convex-nonconvex
        return True

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the polygon.

        >>> Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0)).ndim
        3
        """
        return len(self)

    @property
    def vertices(self) -> List[Point]:
        """Return the vertices of the polygon.

        >>> Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0)).vertices
        [Point(x=0.0, y=0.0), Point(x=1.0, y=0.0), Point(x=1.0, y=1.0)]
        """
        return self.points

    @property
    def edges(self) -> List[Segment]:
        """Return the edges of the polygon.

        >>> edges = Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0)).edges
        >>> edges[0]
        Segment(start=Point(x=0.0, y=0.0), end=Point(x=1.0, y=0.0)) -> Line(A=0.0, B=1.0, C=0.0)
        >>> edges[1]
        Segment(start=Point(x=1.0, y=0.0), end=Point(x=1.0, y=1.0)) -> Line(A=1.0, B=0.0, C=-1.0)
        >>> edges[2]
        Segment(start=Point(x=1.0, y=1.0), end=Point(x=0.0, y=0.0)) -> Line(A=1.0, B=-1.0, C=0.0)
        """
        starts, ends = self.vertices, self.vertices[1:] + self.vertices[:1]
        return [Segment(start=start, end=end) for start, end in zip(starts, ends)]

    def contains(self, item: PointType | Iterable[Point]) -> bool:
        """Check if a point is inside the polygon.

        >>> polygon = Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0))
        >>> polygon.contains((Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)))
        True

        Args:
            item: A point or an iterable of points.
        """
        if isinstance(item, Iterable) and isinstance(tuple(item)[0], Point):
            return all(self.contains(p) for p in item)

        # Lie on the edges or vertices
        point = Point.aspoint(item)
        if self.inVertices(point) or self.inEdges(point):
            return True

        # Ray casting
        ray = Ray(start=point, end=Point(0, 0))
        intersections, intersecting_points = 0, PointSequence()
        for edge in self.edges:
            if ray.isIntersecting(edge) and (intersection := ray.intersect(edge)) not in intersecting_points:
                intersections += 1
                intersecting_points.append(intersection)
        return intersections % 2 == 1

    def inEdges(self, item: PointType | Iterable[Point]) -> bool:
        """Check if a point is on the edges of the polygon.

        >>> polygon = Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0))
        >>> polygon.inEdges((Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)))
        True
        >>> polygon.inEdges((Point(0.5, 0.0), Point(0.5, 1.0), Point(0.0, 0.5), Point(1.0, 0.5)))
        True

        Args:
            item: A point or an iterable of points.
        """
        if isinstance(item, Iterable) and isinstance(tuple(item)[0], Point):
            return all(self.inEdges(p) for p in item)
        point = Point.aspoint(item)
        return any(point in edge for edge in self.edges)

    def inVertices(self, item: PointType | Iterable[Point]) -> bool:
        """Check if a point is on the vertices of the polygon.

        >>> polygon = Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0))
        >>> polygon.inVertices((Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)))
        True

        Args:
            item: A point or an iterable of points.
        """
        if isinstance(item, Iterable) and isinstance(tuple(item)[0], Point):
            return all(self.inVertices(p) for p in item)
        point = Point.aspoint(item)
        return any(point == vertex for vertex in self.vertices)

    @property
    def area(self) -> float:
        """Return the area of the polygon.

        >>> Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)).area
        1.0
        """
        v1, v2 = self.vertices, self.vertices[1:] + self.vertices[:1]
        return abs(sum((p1.x - p2.x) * (p1.y + p2.y) for p1, p2 in zip(v1, v2))) / 2

    @property
    def perimeter(self) -> float:
        """Return the perimeter of the polygon.

        >>> Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)).perimeter
        4.0
        """
        return sum(edge.length for edge in self.edges)

    @property
    def centroid(self) -> Point:
        """Return the centroid of the polygon.

        >>> Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)).centroid
        Point(x=0.5, y=0.5)
        """
        v1, v2 = self.vertices, self.vertices[1:] + self.vertices[:1]
        x = abs(sum((p1.x + p2.x) * (p1.x * p2.y - p2.x * p1.y) for p1, p2 in zip(v1, v2))) / (6 * self.area)
        y = abs(sum((p1.y + p2.y) * (p1.x * p2.y - p2.x * p1.y) for p1, p2 in zip(v1, v2))) / (6 * self.area)
        return Point(x, y)

    @property
    def bounding(self) -> Tuple[Point, Point]:
        """Return the bounding box of the polygon.

        >>> Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)).bounding
        (Point(x=0.0, y=0.0), Point(x=1.0, y=1.0))
        """
        x, y = zip(*self.vertices)
        return Point(min(x), min(y)), Point(max(x), max(y))

    @property
    def internalAngles(self) -> list[float]:
        """Return the internal angles of the polygon.

        >>> Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)).internalAngles
        [90.0, 90.0, 90.0, 90.0]
        """
        return [edge.angle for edge in self.edges]
