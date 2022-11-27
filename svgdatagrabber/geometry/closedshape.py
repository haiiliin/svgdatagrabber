from abc import ABC
from typing import Tuple

from .geometrybase import GeometryBase
from .point import Point


class ClosedShape(GeometryBase, ABC):
    """Base class for closed shapes."""

    def __eq__(self, other: "ClosedShape") -> bool:
        raise NotImplementedError

    def __contains__(self, item: GeometryBase) -> bool:
        """Check if a point or shape is inside the shape."""
        raise NotImplementedError

    def area(self) -> float:
        """Return the area of the shape."""
        raise NotImplementedError

    def perimeter(self) -> float:
        """Return the perimeter of the shape."""
        raise NotImplementedError
    
    def centroid(self) -> Point:
        """Return the centroid of the shape."""
        raise NotImplementedError

    def bounding_box(self) -> Tuple[Point, Point]:
        """Return the bounding box of the shape."""
        raise NotImplementedError
