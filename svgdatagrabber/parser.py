from typing import Union, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from svgpathtools.path import Path, Line, QuadraticBezier, CubicBezier, Arc
from svgpathtools.svg_to_paths import svg2paths

from .filters import (
    RectangleRangeFilter,
    SegmentNumberFilter,
    HorizontalLineFilter,
    VerticalLineFilter,
    FilterBase,
    ClosedPathFilter,
    CustomFilter,
)


class SvgPaths(list[Path]):
    """Svg paths returned by SvgPathParser.parse()"""

    def transformed(
        self,
        translate: tuple[float, float] = (0.0, 0.0),
        rotate: float = 0.0,
        scale: tuple[float, float] = (1.0, 1.0),
        origin: tuple[float, float] = (0.0, 0.0),
    ):
        """Transform the paths.

        Args:
            translate: Translation to apply to the paths.
            rotate: Rotate degree to apply to the paths.
            scale: Scale to apply to the paths.
            origin: Origin to scale the paths around.

        Returns:
            The transformed paths.
        """
        return self.translated(translate).rotated(rotate, origin).scaled(scale, origin)

    def translated(self, translate: tuple[float, float] = (0.0, 0.0)):
        """Translate the paths.

        Args:
            translate: Translation to apply to the paths.

        Returns:
            The translated paths.
        """
        if translate == (0.0, 0.0):
            return self
        for idx, path in enumerate(self):  # type: int, Path
            self[idx] = path.translated(complex(*tuple(translate)))
        return self

    def rotated(self, deg: float = 0.0, origin: tuple[float, float] = (0.0, 0.0)):
        """Rotate the paths.

        Args:
            deg: Rotate degree to apply to the paths.
            origin: Origin to rotate the paths around.

        Returns:
            The rotated paths.
        """
        if deg == 0.0:
            return self
        for idx, path in enumerate(self):
            self[idx] = path.rotated(deg, origin=complex(*tuple(origin)))
        return self

    def scaled(self, scale: tuple[float, float] = (1.0, 1.0), origin: tuple[float, float] = (0.0, 0.0)):
        """Scale the paths.

        Args:
            scale: Scale to apply to the paths.
            origin: Origin to scale the paths around.

        Returns:
            The scaled paths.
        """
        if scale == (1.0, 1.0):
            return self
        for idx, path in enumerate(self):
            self[idx] = path.scaled(*tuple(scale), origin=complex(*tuple(origin)))
        return self

    def lines(self) -> list[np.ndarray]:
        """Get the points of lines from the paths.

        Returns:
            A list of lines of points.
        """
        pathPoints = []
        for path in self:
            points = []
            for segment in path:  # type: Line | QuadraticBezier | CubicBezier | Arc
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
        for point in self.lines():
            ax.plot(point[:, 0], point[:, 1], **kwargs)
        ax.grid()
        return ax

    def df(self, x: str = "x", y: str = "y") -> pd.DataFrame:
        """Get the paths as a pandas DataFrame."""
        df = pd.DataFrame(columns=[x, y, "path"])
        for idx, line in enumerate(self.lines()):
            df = pd.concat([df, pd.DataFrame({x: line[:, 0], y: line[:, 1], "path": idx})], ignore_index=True)
        return df

    def to_csv(self, path: str, x: str = "x", y: str = "y"):
        """Save the paths as a csv file."""
        self.df(x, y).to_csv(path, index=False)


class SvgPathParser:
    """Parse an SVG file and return the paths."""

    #: Path to the svg file
    svgfile: str
    #: Filters
    filters: list[FilterBase]

    def __init__(
        self,
        svgfile: str,
        xrange: tuple[float, float] = (-np.inf, np.inf),
        yrange: tuple[float, float] = (-np.inf, np.inf),
        min_segments: int = 4,
        drop_horizontal_lines: bool = True,
        drop_vertical_lines: bool = True,
        drop_closed_paths: bool = True,
        custom_filter: Union[Callable[[Path], bool], FilterBase] = None,
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
        self.filters = [
            RectangleRangeFilter(xrange=xrange, yrange=yrange),
            SegmentNumberFilter(min_segments=min_segments),
            HorizontalLineFilter(enabled=drop_horizontal_lines, tolerance=tolerance),
            VerticalLineFilter(enabled=drop_vertical_lines, tolerance=tolerance),
            ClosedPathFilter(enabled=drop_closed_paths),
        ]
        self.addFilter(custom_filter)

    def addFilter(self, f: Union[Callable[[Path], bool], FilterBase]):
        """Add a custom filter to the parser.

        Args:
            f: Custom filter for the paths.
        """
        if not f:
            return
        self.filters.append(f if isinstance(f, FilterBase) else CustomFilter(f))

    def parse(
        self,
        translate: tuple[float, float] = (0.0, 0.0),
        rotate: float = 0.0,
        scale: tuple[float, float] = (1.0, 1.0),
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> SvgPaths:
        """Parse the paths from the svg file.

        Args:
            translate: Translation to apply to the paths.
            scale: Scale to apply to the paths.
            rotate: Rotate degree to apply to the paths.
            origin: Origin to scale the paths around.

        Returns:
            A list of paths.
        """

        def filtered(pts: list[Path]) -> list[Path]:
            return [pt for pt in pts if all(f.accept(pt) for f in self.filters)]

        paths, atts = svg2paths(self.svgfile)
        paths = SvgPaths(filtered(paths)).translated(translate).rotated(rotate, origin).scaled(scale, origin)
        return paths
