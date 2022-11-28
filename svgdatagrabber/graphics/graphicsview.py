from __future__ import annotations

from typing import List

from qtpy.QtCore import Qt
from qtpy.QtGui import QResizeEvent, QPainter
from qtpy.QtWidgets import QGraphicsView, QGraphicsScene, QOpenGLWidget

from .geometricobject import GeometricObject
from ..geometry import GeometryBase
from ..geometry.geometrybase import QPenType, QBrushType


class GraphicsView(QGraphicsView):
    """GraphicsView class."""

    #: geometric objects
    geometric_objects: List[GeometricObject]

    def __init__(self, parent=None):
        """Initialize the class."""
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.HighQualityAntialiasing)

        # geometric objects
        self.geometric_objects = []

    def addPrimitive(self, primitive: GeometricObject | GeometryBase):
        """Add a primitive to the scene."""
        if isinstance(primitive, GeometryBase):
            primitive = GeometricObject(primitive)
        primitive.draw(self.scene)
        self.geometric_objects.append(primitive)

    def redraw(self, pen: QPenType = None, brush: QBrushType = None, fit: bool = True):
        """Draw the geometries in the scene and fit the view."""
        for geometric_object in self.geometric_objects:
            geometric_object.draw(self.scene, pen, brush)
        fit and self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event: QResizeEvent):
        """Reimplement the resize event."""
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def useOpenGL(self):
        """Use OpenGL."""
        self.setViewport(QOpenGLWidget())

    def wheelEvent(self, event):
        """
        Zoom in or out of the view, from https://stackoverflow.com/a/29026916/18728919.
        """
        # Zoom factors
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        # Set Anchors
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)

        # Save the scene pos
        oldPos = self.mapToScene(event.pos())

        # Zoom
        zoomFactor = zoomInFactor if event.angleDelta().y() > 0 else zoomOutFactor
        self.scale(zoomFactor, zoomFactor)

        # Get the new position
        newPos = self.mapToScene(event.pos())

        # Move scene to old position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())