from __future__ import annotations

from abc import ABC
from typing import Union, Tuple

from PyQt5.QtCore import QLineF
from PyQt5.QtGui import QPolygonF
from qtpy.QtCore import Qt
from qtpy.QtGui import QPen, QBrush, QColor, QGradient
from qtpy.QtWidgets import QGraphicsScene, QGraphicsItem

QPenType = Union[QPen, QColor, Qt.GlobalColor, QGradient]
QBrushType = Union[QBrush, QColor, Qt.GlobalColor, QGradient]


class GeometryBase(ABC):
    """A base class for all geometries."""

    #: Tolerance for equality.
    tolerance = 1e-6

    def __repr__(self) -> str:
        """Return the representation of the geometry."""
        raise NotImplementedError

    def __eq__(self, other: "GeometryBase") -> bool:
        """Check if two geometries are equal."""
        raise NotImplementedError

    def __ne__(self, other: "GeometryBase") -> bool:
        """Check if two geometries are not equal."""
        return not self.__eq__(other)

    @property
    def drawingargs(self) -> Tuple[str, tuple]:
        """Return the arguments for drawing the geometry."""
        raise NotImplementedError

    def draw(
        self, scene: QGraphicsScene, pen: QPenType = None, brush: QBrushType = None, item: QGraphicsItem = None
    ) -> QGraphicsItem:
        """Draw the geometry on the scene.

        Args:
            scene: The scene to draw on.
            pen: The pen to draw with.
            brush: The brush to draw with.
            item: The old item to draw on, if any.
        """
        args = self.drawingargs
        if isinstance(self, DrawAsLine):
            item and item.setLine(*args) or (item := scene.addLine(*args))
        elif isinstance(self, DrawAsPolygon):
            item and item.setPolygon(*args) or (item := scene.addPolygon(*args))
        elif isinstance(self, DrawAsEllipse):
            item and item.setRect(*args) or (item := scene.addEllipse(*args))
        else:
            raise ValueError(f"Unknown type {type}")
        pen and item.setPen(pen)
        brush and item.setBrush(brush)
        return item


class DrawAsLine(GeometryBase, ABC):
    @property
    def drawingargs(self) -> QLineF:
        raise NotImplementedError


class DrawAsPolygon(GeometryBase, ABC):
    @property
    def drawingargs(self) -> QPolygonF:
        raise NotImplementedError


class DrawAsEllipse(GeometryBase, ABC):
    @property
    def drawingargs(self) -> Tuple[str, Tuple[float, float, float, float]]:
        raise NotImplementedError
