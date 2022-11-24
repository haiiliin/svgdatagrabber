from typing import Union, Callable

import numpy as np
from svgpathtools.path import Path, Line, QuadraticBezier, CubicBezier, Arc


class FilterBase:
    """Base class for filters."""

    #: Enabled or not
    enabled: bool = True
    #: Tolerance for determining if a path is a horizontal or vertical line
    tolerance: float = 1e-6

    def __init__(self, enabled: bool = True, tolerance: float = 1e-6, **kwargs):
        """Constructor of the FilterBase class.

        Args:
            enabled: Enabled or not.
            tolerance: Tolerance for determining if a path is a horizontal or vertical line.
        """
        self.enabled = enabled
        self.tolerance = tolerance
        for key, value in kwargs.items():
            setattr(self, key, value)

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


class RangeFilter(FilterBase):
    """Filter paths based on their range."""

    #: Range of x values to include.
    xrange: tuple[float, float]
    #: Range of y values to include.
    yrange: tuple[float, float]

    def __init__(
        self,
        xrange: tuple[float, float] = (-np.inf, np.inf),
        yrange: tuple[float, float] = (-np.inf, np.inf),
        enabled: bool = True,
        tolerance: float = 1e-6,
    ):
        super().__init__(enabled, tolerance)
        self.xrange = tuple(xrange)
        self.yrange = tuple(yrange)

    def accept(self, path: Path) -> bool:
        return not self.enabled or (
            self.xrange[0] <= path.start.real <= self.xrange[1] and self.yrange[0] <= path.start.imag <= self.yrange[1]
        )


class SegmentNumberFilter(FilterBase):
    """Filter paths based on the number of segments."""

    #: Minimum number of segments in a path.
    min_segments: int

    def __init__(self, min_segments: int = 4, enabled: bool = True, tolerance: float = 1e-6):
        super().__init__(enabled, tolerance)
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

    def __init__(self, filter_function: Callable[[Path], bool], enabled: bool = True, tolerance: float = 1e-6):
        super().__init__(enabled, tolerance)
        self.filter_function = filter_function

    def accept(self, path: Path) -> bool:
        return not self.enabled or self.filter_function(path)
