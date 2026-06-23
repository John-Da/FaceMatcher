import cv2
import sys

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFrame,
)


class WebcamViewer(QWidget):

    revert_camera_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.current_camera = None
        self.flipped = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)

        # ==========================================
        # Webcam Display
        # ==========================================
        self.webcam_label = QLabel("No Camera Connected")
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setMinimumSize(800, 550)
        self.webcam_label.setFrameShape(QFrame.Shape.Box)
        self.webcam_label.setStyleSheet("""
            background-color: #1e1e1e;
            color: white;
            font-size: 18px;
        """)

        # ==========================================
        # Bottom Controls
        # ==========================================
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        self.start_btn = QPushButton("▶ Start Camera")
        self.stop_btn = QPushButton("⏹ Stop Camera")
        self.flip_btn = QPushButton("⇆ Flip")
        self.fps_label = QLabel("FPS: --")

        self.flip_btn.setCheckable(True)
        self.flip_btn.clicked.connect(lambda checked: setattr(self, "flipped", checked))

        for widget in [self.start_btn, self.stop_btn, self.flip_btn]:
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.flip_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.fps_label)

        # ==========================================
        # Add Widgets
        # ==========================================
        main_layout.addWidget(self.webcam_label, 1)
        main_layout.addLayout(controls_layout)

    def start_camera(self, camera_name="0: OBS Virtual Camera"):
        if isinstance(camera_name, bool):
            return

        try:
            index = int(camera_name.split(":")[0].strip())
        except ValueError:
            self.webcam_label.setText("No valid camera selected")
            return

        self.stop_camera()

        if sys.platform == "darwin":
            self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            self.webcam_label.setText(f"Failed to open: {camera_name}")
            self.cap = None
            if self.current_camera:
                self.revert_camera_signal.emit(self.current_camera)
            return

        self.current_camera = camera_name
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.timer.start(int(1000 / fps))

    def stop_camera(self, clear_label=False):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if clear_label:
            self.webcam_label.setText("No Camera Connected")
            self.webcam_label.setPixmap(QPixmap())

    def on_stop_btn(self):
        self.current_camera = None
        self.stop_camera(clear_label=True)

    def next_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        if self.flipped:
            frame = cv2.flip(frame, 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.webcam_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.webcam_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)
