from typing import List

from qtpy.QtGui import QResizeEvent
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QGraphicsView, QGraphicsScene

from .geometricobject import GeometricObject
from ..geometry import Polygon, Point, Circle


class GraphicsView(QGraphicsView):
    """GraphicsView class."""

    #: geometric objects
    geometric_objects: List[GeometricObject]

    def __init__(self, parent=None):
        """Initialize the class."""
        super().__init__(parent)
        self._graphics_scene = QGraphicsScene()
        self.setScene(self._graphics_scene)

        self.geometric_objects = []
        self.geometric_objects.append(
            GeometricObject(Polygon(Point(0.0, 0.0), Point(100.0, 0.0), Point(100.0, 100.0), Point(0.0, 100.0)))
        )
        self.geometric_objects.append(GeometricObject(Circle(center=Point(50.0, 50.0), r=50.0)))
        self.draw()

    def draw(self):
        """Draw the geometries in the scene and fit the view."""
        for geometric_object in self.geometric_objects:
            geometric_object.draw(self._graphics_scene)
        self.fitInView(self._graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Reimplement the resize event."""
        self.fitInView(self._graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
