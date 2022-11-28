from qtpy.QtWidgets import QGraphicsItem, QGraphicsScene

from ..geometry import GeometryBase
from ..geometry.geometrybase import QPenType, QBrushType


class GeometricObject:
    #: The geometry of the object.
    geometry: GeometryBase
    #: The graphics item of the object.
    item: QGraphicsItem

    def __init__(self, geometry: GeometryBase, item: QGraphicsItem = None):
        """Create a geometric object.

        Args:
            geometry: The geometry of the object.
            item: The graphics item of the object.
        """
        self.geometry = geometry
        self.item = item

    def draw(
        self,
        scene: QGraphicsScene,
        pen: QPenType = None,
        brush: QBrushType = None,
    ):
        """Draw the object in the scene."""
        self.item = self.geometry.draw(scene, pen, brush, self.item)
