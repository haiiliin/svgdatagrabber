from qtpy.QtWidgets import QMainWindow, QGraphicsView, QGraphicsScene


class SvgDataGrabberMainWindow(QMainWindow):
    """A custom main window for the data grabber application."""

    def __init__(self, parent=None):
        super(SvgDataGrabberMainWindow, self).__init__(parent)
        self._graphics_view = QGraphicsView()
        self._graphics_scene = QGraphicsScene()
        self._graphics_scene.addText("Hello World")
        self._graphics_view.setScene(self._graphics_scene)
        self.setCentralWidget(self._graphics_view)
