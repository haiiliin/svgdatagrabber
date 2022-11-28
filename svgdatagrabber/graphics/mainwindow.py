from qtpy.QtWidgets import QMainWindow
from .graphicsview import GraphicsView


class SvgDataGrabberMainWindow(QMainWindow):
    """A custom main window for the data grabber application."""

    def __init__(self, parent=None):
        super(SvgDataGrabberMainWindow, self).__init__(parent)
        self._graphics_view = GraphicsView(self)
        self.setCentralWidget(self._graphics_view)

        self.resize(800, 600)
        self.setWindowTitle("SVG Data Grabber")
