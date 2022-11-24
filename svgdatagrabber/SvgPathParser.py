import copy
from typing import Union, Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from svgpathtools.path import Path, Line, QuadraticBezier, CubicBezier, Arc
from svgpathtools.svg_to_paths import svg2paths


class svgdatagrabber(list[Path]):
    """Svg paths returned by SvgPathParser.parse()"""

    def transform(
        self,
        translate: tuple[float, float] = (0.0, 0.0),
        scale: tuple[float, float] = (1.0, 1.0),
        origin: tuple[float, float] = (0.0, 0.0),
    ):
        """Transform the paths.

        Args:
            translate: Translation to apply to the paths.
            scale: Scale to apply to the paths.
            origin: Origin to scale the paths around.

        Returns:
            The transformed paths.
        """
        for idx, path in enumerate(copy.deepcopy(self)):
            translatedPath = path.translated(complex(*translate))
            scaledPath = translatedPath.scaled(*scale, origin=complex(*origin))
            self[idx] = scaledPath
        return self

    def points(self) -> list[np.ndarray]:
        """Get the points from the paths.

        Returns:
            A list of points.
        """
        pathPoints = []
        for path in self:
            points = []
            for segment in path:
                points += [[point.real, point.imag] for point in [segment.start, segment.end]]
            points = np.asarray(points)
            pathPoints.append(points)
        return pathPoints

    def plot(self, ax: Axes = None, fig_kwargs: dict = None, **kwargs) -> Axes:
        """Plot the paths.

        Args:
            ax: Axes to plot on.
            fig_kwargs: Keyword arguments to pass to plt.figure().
            kwargs: Keyword arguments to pass to plt.plot().
        """
        if ax is None:
            _, ax = plt.subplots(**(fig_kwargs or {}))
        for point in self.points():
            ax.plot(point[:, 0], point[:, 1], **kwargs)
        ax.grid()
        return ax


class PathFilter:
    #: Range of x values to include
    xrange: tuple[float, float] = (-np.inf, np.inf)
    #: Range of y values to include
    yrange: tuple[float, float] = (-np.inf, np.inf)
    #: Minimum number of segments in a path
    min_segments: int = 4
    #: Whether to drop horizontal lines
    drop_horizontal_lines: bool = True
    #: Whether to drop closed paths
    drop_vertical_lines: bool = True
    #: Whether to drop closed paths
    drop_closed_paths: bool = True
    #: Custom filter for the paths
    custom_filter: Callable[[Path], bool] = None
    #: Tolerance for determining if a path is a horizontal or vertical line
    tolerance: float = 1e-6

    def pathIsHorizontalLine(self, path: Path):
        def segmentIsHorizontal(segment: Union[Line, QuadraticBezier, CubicBezier, Arc]):
            if type(segment) != Line:
                return False
            return abs(segment.start.imag - segment.end.imag) < self.tolerance

        return all(segmentIsHorizontal(segment) for segment in path)

    def pathIsVerticalLine(self, path: Path):
        def segmentIsVertical(segment: Union[Line, QuadraticBezier, CubicBezier, Arc]):
            if type(segment) != Line:
                return False
            return abs(segment.start.real - segment.end.real) < self.tolerance

        return all(segmentIsVertical(segment) for segment in path)

    def pathIsInRange(self, path: Path):
        def segmentIsInRange(segment: Union[Line, QuadraticBezier, CubicBezier, Arc]):
            for point in segment.bpoints():
                x, y = point.real, point.imag
                if x < self.xrange[0] or x > self.xrange[1] or y < self.yrange[0] or y > self.yrange[1]:
                    return False
            return True

        return all(segmentIsInRange(segment) for segment in path)

    @classmethod
    def pathIsClosed(cls, path: Path):
        return path.iscontinuous() and path.isclosed()

    def includePath(self, path: Path):
        if len(path) < self.min_segments:  # Too few segments
            return False
        if not self.pathIsInRange(path):  # Out of range
            return False
        if self.drop_closed_paths and self.pathIsClosed(path):  # Closed path
            return False
        if self.drop_horizontal_lines and self.pathIsHorizontalLine(path):  # Horizontal line
            return False
        if self.drop_vertical_lines and self.pathIsVerticalLine(path):  # Vertical line
            return False
        if self.custom_filter is not None and not self.custom_filter(path):  # Custom filter
            return False
        return True

    def filter(self, paths: list[Path]) -> list[Path]:
        return [path for path in paths if self.includePath(path)]


class SvgPathParser(PathFilter):
    """Parse an SVG file and return the paths."""

    #: Path to the svg file
    svgfile: str

    def __init__(
        self,
        svgfile: str,
        xrange: tuple[float, float] = (-np.inf, np.inf),
        yrange: tuple[float, float] = (-np.inf, np.inf),
        min_segments: int = 4,
        drop_horizontal_lines: bool = True,
        drop_vertical_lines: bool = True,
        drop_closed_paths: bool = True,
        custom_filter: Callable[[Path], bool] = None,
        tolerance: float = 1e-6,
    ):
        """Constructor of the SvgPathParser class.

        Args:
            svgfile: Path to the svg file.
            xrange: Range of x values to include.
            yrange: Range of y values to include.
            min_segments: Minimum number of segments in a path.
            drop_horizontal_lines: Whether to drop horizontal lines.
            drop_vertical_lines: Whether to drop closed paths.
            drop_closed_paths: Whether to drop closed paths.
            custom_filter: Custom filter for the paths.
            tolerance: Tolerance for determining if a path is a horizontal or vertical line.
        """
        self.svgfile = svgfile
        self.xrange = xrange
        self.yrange = yrange
        self.min_segments = min_segments
        self.drop_horizontal_lines = drop_horizontal_lines
        self.drop_vertical_lines = drop_vertical_lines
        self.drop_closed_paths = drop_closed_paths
        self.custom_filter = custom_filter
        self.tolerance = tolerance

    def parse(
        self,
        translate: tuple[float, float] = (0.0, 0.0),
        scale: tuple[float, float] = (1.0, 1.0),
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> svgdatagrabber:
        """Parse the paths from the svg file.

        Args:
            translate: Translation to apply to the paths.
            scale: Scale to apply to the paths.
            origin: Origin to scale the paths around.

        Returns:
            A list of paths.
        """
        paths = svgdatagrabber(self.filter(svg2paths(self.svgfile)[0]))
        paths.transform(tuple(translate), tuple(scale), tuple(origin))
        return paths
