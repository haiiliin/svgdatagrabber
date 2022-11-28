from abc import ABC
from typing import Union

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
    def qobject(self):
        """Return the geometry as a Qt object."""
        raise NotImplementedError

    def draw(self, scene: QGraphicsScene, pen: QPenType = None, brush: QBrushType = None) -> QGraphicsItem:
        """Draw the geometry in the scene."""
        raise NotImplementedError
