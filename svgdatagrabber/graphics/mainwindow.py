from typing import List

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsItem

from .geometricobject import GeometricObject
from ..geometry import Polygon, Point, Circle


class SvgDataGrabberMainWindow(QMainWindow):
    """A custom main window for the data grabber application."""

    #: geometric objects
    geometric_objects: List[GeometricObject]

    def __init__(self, parent=None):
        super(SvgDataGrabberMainWindow, self).__init__(parent)
        self._graphics_view = QGraphicsView()
        self._graphics_scene = QGraphicsScene()
        self._graphics_view.setScene(self._graphics_scene)
        self.setCentralWidget(self._graphics_view)

        self.geometric_objects = []
        self.geometric_objects.append(
            GeometricObject(Polygon(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)))
        )
        self.geometric_objects.append(GeometricObject(Circle(center=Point(0.5, 0.5), r=0.5)))
        self.draw()

    def draw(self):
        """Draw the geometries in the scene and fit the view."""
        for geometric_object in self.geometric_objects:
            geometric_object.draw(self._graphics_scene)
        self.fitInView(self._graphics_scene.itemsBoundingRect())

    def fitInView(self, item: QGraphicsItem, aspectRadioMode=Qt.KeepAspectRatio):
        self._graphics_view.fitInView(item, aspectRadioMode)
