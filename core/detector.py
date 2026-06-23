import os
import cv2
import numpy as np
from ultralytics import YOLO

MODELS_PATH = "data/models/detectors"
DETECTOR_CONF = 0.4  # internal confidence, not exposed to UI


class Detector:

    def __init__(self, model_name: str):
        model_path = os.path.join(MODELS_PATH, model_name)
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run person detection on a single frame.

        Returns:
            List of dicts: [{ "box": (x1, y1, x2, y2), "crop": np.ndarray, "conf": float }]
        """
        results = self.model(frame, conf=DETECTOR_CONF, verbose=False)[0]
        detections = []

        for box in results.boxes:
            if int(box.cls[0]) != 0:  # person only
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            detections.append(
                {
                    "box": (x1, y1, x2, y2),
                    "crop": crop,
                    "conf": conf,
                }
            )

        return detections

    def detect_single(self, image: np.ndarray) -> np.ndarray | None:
        """
        For reference image: detect the first person and return their crop.
        Returns None if no person is found.
        """
        detections = self.detect(image)
        if not detections:
            return None
        # take highest confidence detection
        best = max(detections, key=lambda d: d["conf"])
        return best["crop"]
