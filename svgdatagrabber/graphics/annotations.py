from typing import Union

from qtpy.QtCore import Qt
from qtpy.QtGui import QPen, QBrush, QColor, QGradient
from qtpy.QtWidgets import QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsEllipseItem

QPenType = Union[QPen, QColor, Qt.GlobalColor, QGradient]
QBrushType = Union[QBrush, QColor, Qt.GlobalColor, QGradient]
QGraphicsItemType = Union[QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsEllipseItem]
