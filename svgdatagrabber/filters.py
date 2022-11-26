from __future__ import annotations

from typing import Union, Callable

import numpy as np
from svgpathtools.path import Path, Line, QuadraticBezier, CubicBezier, Arc


class FilterBase:
    """Base class for filters."""

    #: Enabled or not
    enabled: bool = True
    #: Tolerance for determining if a path is a horizontal or vertical line
    tolerance: float = 1e-6

    def __init__(self, *, enabled: bool = True, tolerance: float = 1e-6):
        self.enabled = enabled
        self.tolerance = tolerance

    def accept(self, path: Path) -> bool:
        """Accept or reject a path.

        Args:
            path: Path to check.

        Returns:
            True if the path is accepted, False otherwise.
        """
        raise NotImplementedError

    def __call__(self, path: Path) -> bool:
        return self.accept(path)

    def enable(self):
        """Enable the filter."""
        self.enabled = True

    def disable(self):
        """Disable the filter."""
        self.enabled = False


class RectangleRangeFilter(FilterBase):
    """Filter paths based on their range."""

    #: Range of x values to include.
    xrange: tuple[float, float]
    #: Range of y values to include.
    yrange: tuple[float, float]
    #: Include or exclude the range
    include: bool
    #: Sensitive or not
    sensitive: bool

    def __init__(
        self,
        xrange: tuple[float, float] = (-np.inf, np.inf),
        yrange: tuple[float, float] = (-np.inf, np.inf),
        include: bool = True,
        sensitive: bool = False,
        *,
        enabled: bool = True,
        tolerance: float = 1e-6,
    ):
        super().__init__(enabled=enabled, tolerance=tolerance)
        self.xrange = tuple(xrange)
        self.yrange = tuple(yrange)
        self.include = include
        self.sensitive = sensitive

    def accept(self, path: Path) -> bool:
        def pointInRange(p: complex):
            return self.xrange[0] <= p.real <= self.xrange[1] and self.yrange[0] <= p.imag <= self.yrange[1]

        def segmentInRange(seg: Union[Line, QuadraticBezier, CubicBezier, Arc]):
            func = all if self.sensitive else any
            return func(pointInRange(p) for p in seg.bpoints())

        def pathInRange(pth: Path):
            func = all if self.sensitive else any
            return func(segmentInRange(seg) for seg in pth)

        def pathOutRange(pth: Path):
            return not pathInRange(pth)

        return not self.enabled or (self.include and pathInRange(path)) or (not self.include and pathOutRange(path))


class SegmentNumberFilter(FilterBase):
    """Filter paths based on the number of segments."""

    #: Minimum number of segments in a path.
    min_segments: int

    def __init__(self, min_segments: int = 4, *, enabled: bool = True, tolerance: float = 1e-6):
        super().__init__(enabled=enabled, tolerance=tolerance)
        self.min_segments = min_segments

    def accept(self, path: Path) -> bool:
        return not self.enabled or len(path) >= self.min_segments


class HorizontalLineFilter(FilterBase):
    """Filter for horizontal lines."""

    def accept(self, path: Path) -> bool:
        def segmentIsHorizontal(segment: Union[Line, QuadraticBezier, CubicBezier, Arc]):
            if type(segment) != Line:
                return False
            return abs(segment.start.imag - segment.end.imag) < self.tolerance

        return not self.enabled or not (all(segmentIsHorizontal(segment) for segment in path))


class VerticalLineFilter(FilterBase):
    """Filter for vertical lines."""

    def accept(self, path: Path) -> bool:
        def segmentIsVertical(segment: Union[Line, QuadraticBezier, CubicBezier, Arc]):
            if type(segment) != Line:
                return False
            return abs(segment.start.real - segment.end.real) < self.tolerance

        return not self.enabled or not (all(segmentIsVertical(segment) for segment in path))


class ClosedPathFilter(FilterBase):
    """Filter for closed paths."""

    def accept(self, path: Path) -> bool:
        return not self.enabled or not (path.iscontinuous() and path.isclosed())


class CustomFilter(FilterBase):
    """Custom filter."""

    #: Custom filter function
    filter_function: Callable[[Path], bool]

    def __init__(self, filter_function: Callable[[Path], bool], *, enabled: bool = True, tolerance: float = 1e-6):
        super().__init__(enabled=enabled, tolerance=tolerance)
        self.filter_function = filter_function

    def accept(self, path: Path) -> bool:
        return not self.enabled or self.filter_function(path)


def test():
    import doctest
    doctest.testmod()
