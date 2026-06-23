import cv2
import sys
import time
from queue import Queue, Full
from PySide6.QtCore import QThread, Signal


class FrameReader(QThread):

    error = Signal(str)
    finished = Signal()
    fps_ready = Signal(float, int)  # emits (fps, total_frames) once known

    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue
        self.source = None
        self.source_path = None
        self._running = False
        self.fps = 30.0

    def setup(self, source: str, source_path: str):
        self.source = source
        self.source_path = source_path

    def run(self):
        self._running = True
        cap = self._open_capture()
        if cap is None:
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_ready.emit(self.fps, total_frames)

        frame_interval = 1.0 / self.fps

        while self._running:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            try:
                self.queue.put(frame, timeout=2.0)
            except Full:
                continue

            # Only throttle for webcam — video feeds as fast as processor handles it
            if self.source == "webcam":
                elapsed = time.time() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        cap.release()
        self.finished.emit()

    def stop(self):
        self._running = False

    def _open_capture(self):
        if self.source == "webcam":
            index = int(str(self.source_path).split(":")[0].strip())
            cap = (
                cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
                if sys.platform == "darwin"
                else cv2.VideoCapture(index)
            )
        else:
            cap = cv2.VideoCapture(self.source_path)

        if not cap.isOpened():
            self.error.emit(f"Cannot open source: {self.source_path}")
            return None
        return cap
