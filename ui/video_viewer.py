import cv2
import numpy as np
from collections import deque
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QFrame,
)


class VideoViewer(QWidget):

    def __init__(self):
        super().__init__()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.total_frames = 0
        self.total_duration = 0.0
        self.fps = 30.0
        self.is_seeking = False
        self.is_playing = False

        # Processed buffer playback
        self._mode = "file"  # "file" | "buffer" | "processing"
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_index = 0
        self._buffer_snapshot: list[np.ndarray] = []
        self._overlay_frame: np.ndarray | None = None

        # Processing-mode state
        self._processing_total_frames = 0

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)

        self.video_label = QLabel("No Video Loaded")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 400)
        self.video_label.setFrameShape(QFrame.Shape.Box)
        self.video_label.setStyleSheet("""
            background-color: #1e1e1e;
            color: white;
            font-size: 18px;
        """)

        timeline_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.pause_btn = QPushButton("⏸ Pause")
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(1000)
        self.timeline_slider.sliderPressed.connect(self.on_slider_pressed)
        self.timeline_slider.sliderReleased.connect(self.on_slider_released)

        self.time_label = QLabel("0:00 / 0:00")
        self.fps_label = QLabel("FPS: 0")

        timeline_layout.addWidget(self.play_btn)
        timeline_layout.addWidget(self.pause_btn)
        timeline_layout.addWidget(self.timeline_slider, 3)
        timeline_layout.addWidget(self.time_label)
        timeline_layout.addWidget(self.fps_label)

        for w in [self.play_btn, self.pause_btn, self.timeline_slider]:
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        main_layout.addWidget(self.video_label, 1)
        main_layout.addLayout(timeline_layout)

    # --------------------------------------------------
    # Normal file mode
    # --------------------------------------------------
    def load_video(self, file_path: str):
        self._reset()
        self._mode = "file"

        if self.cap:
            self.cap.release()
        self.timer.stop()

        self.cap = cv2.VideoCapture(file_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_duration = self.total_frames / self.fps
        self.is_playing = False

        self.timeline_slider.setValue(0)
        self.fps_label.setText(f"FPS: {self.fps:.1f}")
        self.time_label.setText(f"0:00 / {self._format_time(self.total_duration)}")

        ret, frame = self.cap.read()
        if ret:
            self._display_bgr(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --------------------------------------------------
    # Processing mode — live annotated frames come in via display_live_frame()
    # --------------------------------------------------
    def enter_processing_mode(self, fps: float, total_frames: int):
        self._reset()
        self._mode = "processing"
        self.fps = fps
        self.total_frames = total_frames
        self._processing_total_frames = total_frames
        self.total_duration = total_frames / fps if fps else 0

        self._overlay_frame = None
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.play()

        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.time_label.setText(f"0:00 / {self._format_time(self.total_duration)}")
        self.timeline_slider.setValue(0)
        self.video_label.setText("Processing…")

    def display_live_frame(self, frame: np.ndarray, frame_num: int):
        """
        Called by MainWindow.on_frame_ready() for every annotated frame
        while processing is running. Shows it immediately and updates the
        timeline so the user sees progress in real time.
        """
        self._overlay_frame = frame
        if self._mode != "processing":
            return
        self._display_bgr(frame)
        current_sec = frame_num / self.fps if self.fps else 0
        self._update_timeline(current_sec)

    def stop_processing_mode(self):
        """
        Called by stop_processing() — user hit Stop before finishing.
        Re-enables controls and falls back to file mode if a file is open,
        otherwise clears to idle.
        """
        self._reset()
        self._overlay_frame = None
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.timeline_slider.setEnabled(True)

        if self.cap and self.cap.isOpened():
            # Rewind to start so file is still usable
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._mode = "file"
            ret, frame = self.cap.read()
            if ret:
                self._display_bgr(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self._mode = "file"
            self.video_label.clear()
            self.video_label.setText("No Video Loaded")

    # --------------------------------------------------
    # Buffer mode — called after processing finishes
    # --------------------------------------------------
    def load_buffer(self, buffer: deque, fps: float):
        """Load processed frames and play them like a video."""
        self._reset()
        self._mode = "buffer"

        # Re-enable controls that were locked during processing
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.timeline_slider.setEnabled(True)

        self._buffer_snapshot = list(buffer)
        self._buffer_index = 0
        self.fps = fps
        self.total_frames = len(self._buffer_snapshot)
        self.total_duration = self.total_frames / fps if fps else 0

        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.time_label.setText(f"0:00 / {self._format_time(self.total_duration)}")
        self.timeline_slider.setValue(0)

        if self._buffer_snapshot:
            self._display_bgr(self._buffer_snapshot[0])

        # Auto-play the result
        self.play()

    def reset_to_file_mode(self):
        """Called when Stop Processing is clicked — go back to plain file."""
        self._reset()
        self._mode = "file"
        self.video_label.clear()
        self.video_label.setText("No Video Loaded")

    # --------------------------------------------------
    # Timer tick (file + buffer playback only)
    # --------------------------------------------------
    def _tick(self):
        if self._mode == "buffer":
            self._tick_buffer()
        else:
            self._tick_file()

    def _tick_buffer(self):
        if self._buffer_index >= len(self._buffer_snapshot):
            self.timer.stop()
            self.is_playing = False
            return

        frame = self._buffer_snapshot[self._buffer_index]
        self._display_bgr(frame)

        current_sec = self._buffer_index / self.fps
        self._update_timeline(current_sec)
        self._buffer_index += 1

    def set_overlay_frame(self, frame: np.ndarray | None):
        """During processing, show this annotated frame instead of the file frame."""
        self._overlay_frame = frame

    def _tick_file(self):
        if not self.cap or self.is_seeking:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.is_playing = False
            return

        current_sec = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self._update_timeline(current_sec)

        # Show annotated overlay if processing, otherwise raw frame
        display = getattr(self, "_overlay_frame", None)
        self._display_bgr(display if display is not None else frame)

    # --------------------------------------------------
    # Controls
    # --------------------------------------------------
    def play(self):
        if self._mode == "buffer" and not self._buffer_snapshot:
            return
        if self._mode == "file" and not self.cap:
            return
        if self._mode == "processing":
            return  # nothing to play-back yet
        self.is_playing = True
        self.timer.start(int(1000 / max(self.fps, 1)))

    def pause(self):
        self.is_playing = False
        self.timer.stop()

    def on_slider_pressed(self):
        self.is_seeking = True
        self.timer.stop()

    def on_slider_released(self):
        if self._mode == "buffer":
            pos = int(
                (self.timeline_slider.value() / 1000.0) * len(self._buffer_snapshot)
            )
            self._buffer_index = min(pos, len(self._buffer_snapshot) - 1)
            if self._buffer_snapshot:
                self._display_bgr(self._buffer_snapshot[self._buffer_index])
                self._update_timeline(self._buffer_index / self.fps)
        else:
            if self.cap and self.total_duration:
                seek_sec = (self.timeline_slider.value() / 1000.0) * self.total_duration
                self.cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec * 1000)
                ret, frame = self.cap.read()
                if ret:
                    self._display_bgr(frame)
                    self._update_timeline(seek_sec)

        self.is_seeking = False
        if self.is_playing:
            self.timer.start(int(1000 / max(self.fps, 1)))

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _reset(self):
        self.timer.stop()
        self.is_playing = False
        self._buffer_snapshot = []
        self._buffer_index = 0

    def _display_bgr(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def _update_timeline(self, current_sec: float):
        if self.total_duration:
            self.timeline_slider.setValue(
                int((current_sec / self.total_duration) * 1000)
            )
        self.time_label.setText(
            f"{self._format_time(current_sec)} / {self._format_time(self.total_duration)}"
        )

    def _format_time(self, seconds: float) -> str:
        seconds = int(seconds)
        h, r = divmod(seconds, 3600)
        m, s = divmod(r, 60)
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"
