from PySide6.QtCore import QThread, Signal

from core.detector import Detector
from core.reid import load_reid

class ModelLoader(QThread):

    status = Signal(str)  # update loading dialog text
    finished = Signal(object, object)  # (detector, reid)
    error = Signal(str)

    def __init__(self, detector_name: str, reid_name: str):
        super().__init__()
        self.detector_name = detector_name
        self.reid_name = reid_name

    def run(self):
        try:
            self.status.emit("Loading detector...")
            detector = Detector(self.detector_name)

            self.status.emit("Loading ReID model...")
            reid = load_reid(self.reid_name)  # ← auto picks OSNet or DINOv2

            self.finished.emit(detector, reid)
        except Exception as e:
            self.error.emit(str(e))
