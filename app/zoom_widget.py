from PyQt5 import QtWidgets, QtCore, QtGui

class ZoomableImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QtGui.QPixmap()
        self.zoom_factor = 1.0
        self.pan_offset = QtCore.QPoint()
        self.last_mouse_pos = QtCore.QPoint()
        self._is_first_paint = True

        self._navigation_mode = False
        self._navigation_callback = None

        self.setCursor(QtCore.Qt.OpenHandCursor)

    def setPixmap(self, pixmap):
        """Set the pixmap to be displayed, preserving the current view unless it's the first paint."""
        self.pixmap = pixmap
        self.update()

    def fitToWindow(self):
        """Fit the image to the window, preserving aspect ratio. Can be called to reset the view."""
        if self.pixmap.isNull():
            return
        
        x_ratio = self.width() / self.pixmap.width()
        y_ratio = self.height() / self.pixmap.height()
        self.zoom_factor = min(x_ratio, y_ratio)
        self.pan_offset = QtCore.QPoint()
        self.update()

    def enable_navigation_mode(self, enabled: bool):
        """Enable or disable navigation mode."""
        self._navigation_mode = enabled
        self.setCursor(QtCore.Qt.CrossCursor if enabled else QtCore.Qt.OpenHandCursor)

    def set_navigation_callback(self, callback):
        """Set the callback function for navigation target selection."""
        self._navigation_callback = callback

    def wheelEvent(self, event: QtGui.QWheelEvent):
        """Handle mouse wheel event for zooming."""
        self._is_first_paint = False  # Manual zoom overrides any initial fit
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # Zoom
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        
        self.zoom_factor *= zoom_factor
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """Handle mouse press event for panning or navigation."""
        if self._navigation_mode and event.button() == QtCore.Qt.LeftButton:
            # Handle navigation mode click
            if not self.pixmap.isNull():
                pixmap_size = self.pixmap.size()
                scaled_pixmap_size = pixmap_size * self.zoom_factor

                # Calculate the top-left corner of the scaled pixmap
                x_offset = (self.width() - scaled_pixmap_size.width()) / 2 + self.pan_offset.x()
                y_offset = (self.height() - scaled_pixmap_size.height()) / 2 + self.pan_offset.y()

                # Map the click position to the original image's coordinates
                pixel_x = (event.pos().x() - x_offset) / self.zoom_factor
                pixel_y = (event.pos().y() - y_offset) / self.zoom_factor

                # Ensure the coordinates are within the image bounds
                if 0 <= pixel_x < pixmap_size.width() and 0 <= pixel_y < pixmap_size.height():
                    if self._navigation_callback:
                        self._navigation_callback(int(pixel_x), int(pixel_y))
                else:
                    print("Click out of bounds.")
            self.enable_navigation_mode(False)  # Exit navigation mode after selection
        elif event.button() == QtCore.Qt.LeftButton:
            # Handle panning
            self._is_first_paint = False  # Manual pan overrides any initial fit
            self.last_mouse_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        """Handle mouse move event for panning."""
        if not self._navigation_mode and event.buttons() == QtCore.Qt.LeftButton:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """Handle mouse release event to stop panning."""
        if not self._navigation_mode and event.button() == QtCore.Qt.LeftButton:
            self.setCursor(QtCore.Qt.OpenHandCursor)

    def paintEvent(self, event: QtGui.QPaintEvent):
        """Paint the pixmap with the current zoom and pan."""
        if self.pixmap.isNull():
            return

        # On the first paint event, fit the image to the window.
        if self._is_first_paint:
            self.fitToWindow()
            self._is_first_paint = False

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        # Calculate the target rectangle to draw the pixmap
        pixmap_size = self.pixmap.size()
        scaled_pixmap_size = pixmap_size * self.zoom_factor
        
        # The top-left corner of the scaled pixmap
        x = (self.width() - scaled_pixmap_size.width()) / 2 + self.pan_offset.x()
        y = (self.height() - scaled_pixmap_size.height()) / 2 + self.pan_offset.y()
        
        target_rect = QtCore.QRectF(x, y, scaled_pixmap_size.width(), scaled_pixmap_size.height())
        source_rect = QtCore.QRectF(self.pixmap.rect())

        painter.drawPixmap(target_rect, self.pixmap, source_rect)

    def sizeHint(self) -> QtCore.QSize:
        """Provide a size hint for the layout manager."""
        if self.pixmap.isNull():
            return QtCore.QSize(400, 300)
        # Base the hint on the pixmap's size, but don't be excessively large
        return self.pixmap.size().scaled(400, 300, QtCore.Qt.KeepAspectRatio)

    def minimumSizeHint(self) -> QtCore.QSize:
        """Provide a minimum size hint."""
        return QtCore.QSize(200, 100)
