from __future__ import annotations

import sys
from abc import ABC
from enum import IntEnum
from typing import Union, Tuple

from qtpy.QtCore import Qt
from qtpy.QtGui import QPen, QBrush, QColor, QGradient
from qtpy.QtWidgets import QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsEllipseItem, QApplication
from qtpy.QtWidgets import QGraphicsScene

QPenType = Union[QPen, QColor, Qt.GlobalColor, QGradient]
QBrushType = Union[QBrush, QColor, Qt.GlobalColor, QGradient]
QGraphicsItemType = Union[QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsEllipseItem]


class GeometryDrawAs(IntEnum):
    """Geometry draw as enum."""

    DrawAsLine = 0
    DrawAsPolygon = 1
    DrawAsEllipse = 2


DrawAsLine = GeometryDrawAs.DrawAsLine
DrawAsPolygon = GeometryDrawAs.DrawAsPolygon
DrawAsEllipse = GeometryDrawAs.DrawAsEllipse


class GeometryBase(ABC):
    """A base class for all geometries."""

    #: Tolerance for equality.
    tolerance = 1e-6

    #: type of drawing
    drawAs: GeometryDrawAs

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
    def drawArgs(self) -> Tuple[str, tuple]:
        """Return the arguments for drawing the geometry."""
        raise NotImplementedError

    def draw(
        self,
        scene: QGraphicsScene,
        pen: QPenType = None,
        brush: QBrushType = None,
        item: QGraphicsItemType = None,
    ) -> QGraphicsItemType:
        """Draw the geometry on the scene.

        Args:
            scene: The scene to draw on.
            pen: The pen to draw with.
            brush: The brush to draw with.
            item: The old item to draw on, if any.
        """
        args = self.drawArgs
        if self.drawAs == DrawAsLine:
            item and item.setLine(*args) or item or (item := scene.addLine(*args))
        elif self.drawAs == DrawAsPolygon:
            item and item.setPolygon(*args) or item or (item := scene.addPolygon(*args))
        elif self.drawAs == DrawAsEllipse:
            item and item.setRect(*args) or item or (item := scene.addEllipse(*args))
        else:
            raise ValueError(f"Unknown type {self.drawAs}")
        pen and item.setPen(pen)
        brush and item.setBrush(brush)
        return item

    def plot(self):
        """Plot the geometry."""
        from ..graphics.graphicsview import GraphicsView

        app = QApplication(sys.argv)
        scene = QGraphicsScene()
        view = GraphicsView(scene)
        view.setWindowTitle(repr(self))
        view.resize(800, 600)
        view.show()
        self.draw(scene, Qt.blue, Qt.red)
        sys.exit(app.exec_())
