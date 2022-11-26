from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from svgpathtools.path import Path, Line, QuadraticBezier, CubicBezier, Arc

from .csys import CoordinateSystem


class SvgPaths(List[Path]):
    """Svg paths returned by SvgPathParser.parse()"""

    def transformed(self, csys: CoordinateSystem):
        """Transform the paths.

        Args:
            csys: Coordinate system to transform the paths to.

        Returns:
            The transformed paths.
        """
        for path in self:
            for segment in path:  # type: Line | QuadraticBezier | CubicBezier | Arc
                segment.start = csys.transform(segment.start)
                segment.end = csys.transform(segment.end)
                if isinstance(segment, QuadraticBezier):
                    segment.control = csys.transform(segment.control)
                elif isinstance(segment, CubicBezier):
                    segment.control1 = csys.transform(segment.control1)
                    segment.control2 = csys.transform(segment.control2)
        return self

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
        lines = self.lines()
        for idx, point in enumerate(lines):
            kwargs["label"] = f"Line-{idx}"
            ax.plot(point[:, 0], point[:, 1], **kwargs)
        ax.legend(ncol=math.floor(math.sqrt(len(lines))))
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
