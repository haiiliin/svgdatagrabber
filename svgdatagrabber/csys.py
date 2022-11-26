from __future__ import annotations

from typing import Iterable

import numpy as np

from .geometry import Segment, Point


class Axis(Segment):
    pass


class XAxis(Axis):
    #: X value of the first point.
    xstart: float
    #: X value of the second point.
    xend: float
    #: Y value of the axis.
    y: float

    def __init__(
        self,
        *,
        start: Point | Iterable[float] | complex = Point(0.0, 0.0),
        end: Point | Iterable[float] | complex = Point(1.0, 0.0),
        xstart: float = 0.0,
        xend: float = 1.0,
        y: float = 0.0,
    ):
        super().__init__(start=start, end=end)
        self.xstart, self.xend = xstart, xend
        self.y = y

    def setup(
        self,
        *,
        start: Point | Iterable[float] | complex = Point(0.0, 0.0),
        end: Point | Iterable[float] | complex = Point(1.0, 0.0),
        xstart: float = 0.0,
        xend: float = 1.0,
        y: float = 0.0,
    ):
        """Set up the axis.

        Args:
            start: The first point to create the line.
            end: The second point to create the line.
            xstart: X value of the first point.
            xend: X value of the second point.
            y: Y value of the axis.
        """
        start, end = Point.aspoint(start), Point.aspoint(end)
        self.A, self.B, self.C = self.coefficientsFromTwoPoints(start, end)
        self.start, self.end = start, end
        self.xstart, self.xend = xstart, xend
        self.y = y


class YAxis(Axis):
    #: Y value of the first point.
    ystart: float
    #: Y value of the second point.
    yend: float
    #: X value of the axis.
    x: float

    def __init__(
        self,
        *,
        start: Point | Iterable[float] | complex = Point(0.0, 0.0),
        end: Point | Iterable[float] | complex = Point(0.0, 1.0),
        ystart: float = 0.0,
        yend: float = 1.0,
        x: float = 0.0,
    ):
        super().__init__(start=start, end=end)
        self.ystart, self.yend = ystart, yend
        self.x = x

    def setup(
        self,
        *,
        start: Point | Iterable[float] | complex = Point(0.0, 0.0),
        end: Point | Iterable[float] | complex = Point(0.0, 1.0),
        ystart: float = 0.0,
        yend: float = 1.0,
        x: float = 0.0,
    ):
        """Set up the axis.

        Args:
            start: The first point to create the line.
            end: The second point to create the line.
            ystart: Y value of the first point.
            yend: Y value of the second point.
            x: X value of the axis.
        """
        start, end = Point.aspoint(start), Point.aspoint(end)
        self.A, self.B, self.C = self.coefficientsFromTwoPoints(start, end)
        self.start, self.end = start, end
        self.ystart, self.yend = ystart, yend
        self.x = x


class CoordinateSystem:
    #: The x-axis of the coordinate system.
    xaxis: XAxis
    #: The y-axis of the coordinate system.
    yaxis: YAxis

    def __init__(self):
        self.xaxis = XAxis()
        self.yaxis = YAxis()

    def transform(self, p: Point | Iterable[float] | complex) -> Point:
        """Transform the coordinate to the coordinate system.

        Args:
            p: The point to convert.

        Returns:
            The coordinate in the coordinate system.
        """
        xp = self.yaxis.parallel(p).intersect(self.xaxis)
        yp = self.xaxis.parallel(p).intersect(self.yaxis)
        x = (xp.x - self.xaxis.start.x) / (self.xaxis.end.x - self.xaxis.start.x)
        y = (yp.y - self.yaxis.start.y) / (self.yaxis.end.y - self.yaxis.start.y)
        return Point(x, y)

    def transformation_matrix(self) -> np.ndarray:
        """Get the transformation matrix.

        Returns:
            The transformation matrix.
        """
        return np.array(
            [
                [self.xaxis.end.x - self.xaxis.start.x, 0.0, self.xaxis.start.x],
                [0.0, self.yaxis.end.y - self.yaxis.start.y, self.yaxis.start.y],
                [0.0, 0.0, 1.0],
            ]
        )

    def setup_xaxis(
        self,
        *,
        start: Point | Iterable[float] | complex = Point(0.0, 0.0),
        end: Point | Iterable[float] | complex = Point(1.0, 0.0),
        xstart: float = 0.0,
        xend: float = 1.0,
        y: float = 0.0,
        perpendicular: bool = False,
    ):
        """Set up the x-axis.

        Args:
            start: The first point to create the line.
            end: The second point to create the line.
            xstart: X value of the first point.
            xend: X value of the second point.
            y: Y value of the axis.
            perpendicular: If the axis should be perpendicular to the y-axis.
        """
        start, end = Point.aspoint(start), Point.aspoint(end)
        if perpendicular:
            end = self.yaxis.perpendicular(start).intersect(self.yaxis.parallel(end))
        self.xaxis.setup(start=start, end=end, xstart=xstart, xend=xend, y=y)
        if not self.xaxis.isPerpendicular(self.yaxis):
            raise ValueError("The x-axis and y-axis must be perpendicular.")

    def setup_yaxis(
        self,
        *,
        start: Point | Iterable[float] | complex = Point(0.0, 0.0),
        end: Point | Iterable[float] | complex = Point(0.0, 1.0),
        ystart: float = 0.0,
        yend: float = 1.0,
        x: float = 0.0,
        perpendicular: bool = False,
    ):
        """Set up the y-axis.

        Args:
            start: The first point to create the line.
            end: The second point to create the line.
            ystart: Y value of the first point.
            yend: Y value of the second point.
            x: X value of the axis.
            perpendicular: If the axis should be perpendicular to the y-axis.
        """
        start, end = Point.aspoint(start), Point.aspoint(end)
        if perpendicular:
            end = self.xaxis.perpendicular(start).intersect(self.xaxis.parallel(end))
        self.yaxis.setup(start=start, end=end, ystart=ystart, yend=yend, x=x)
        if not self.yaxis.isPerpendicular(self.xaxis):
            raise ValueError("The x-axis and y-axis must be perpendicular.")