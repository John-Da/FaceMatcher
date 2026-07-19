import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QFrame,
)


class ImageViewer(QWidget):

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 550)
        self.image_label.setFrameShape(QFrame.Shape.Box)
        self.image_label.setStyleSheet("""
            background-color: #1e1e1e;
            color: white;
            font-size: 18px;
        """)

        layout.addWidget(self.image_label)

    def set_status(self, text: str):
        """Show a temporary text overlay (e.g. 'Processing...')"""
        self.image_label.setText(text)

    def load_image(self, file_path: str):
        """Display raw image from file path (before processing)."""
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.image_label.setText("Failed to load image")
            return
        self._set_pixmap(pixmap)

    def display_frame(self, frame: np.ndarray):
        """Display annotated numpy frame (BGR) from processor."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._set_pixmap(QPixmap.fromImage(image))

    def _set_pixmap(self, pixmap: QPixmap):
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setText("")
