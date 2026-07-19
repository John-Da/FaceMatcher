import os
import numpy as np
import torch
from ultralytics import YOLO

MODELS_PATH = "data/models/detectors"

DETECTOR_CONF = 0.15
DETECTOR_IOU = 0.7
DETECTOR_IMGSZ = 1280
DETECTOR_MAX_DET = 300
MIN_PERSON_HEIGHT = 80


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Detector:

    def __init__(self, model_name: str):
        model_path = os.path.join(MODELS_PATH, model_name)
        self.device = get_device()
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(
            f"[Detector] Loaded on device: {next(self.model.model.parameters()).device}"
        )

    def detect(self, frame: np.ndarray, imgsz: int | None = None) -> list[dict]:
        """
        Run person detection on a single frame.

        Returns:
            List of dicts:
            [
                {
                    "box": (x1, y1, x2, y2),
                    "crop": np.ndarray,
                    "conf": float
                }
            ]
        """

        results = self.model(
            frame,
            classes=[0],
            conf=DETECTOR_CONF,
            iou=DETECTOR_IOU,
            imgsz=imgsz or DETECTOR_IMGSZ,
            max_det=DETECTOR_MAX_DET,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        h, w = frame.shape[:2]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            box_h = y2 - y1
            if box_h < MIN_PERSON_HEIGHT:
                continue

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            detections.append(
                {
                    "box": (x1, y1, x2, y2),
                    "crop": crop,
                    "conf": float(box.conf[0]),
                }
            )

        return detections

    def detect_single(self, image: np.ndarray) -> np.ndarray | None:
        """
        For reference image:
        detect the highest-confidence person and return their crop.
        """

        detections = self.detect(image)
        if not detections:
            return None

        best = max(detections, key=lambda d: d["conf"])
        return best["crop"]
