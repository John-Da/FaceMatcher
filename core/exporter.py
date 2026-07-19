import cv2
import numpy as np
import csv
import os


class Exporter:

    @staticmethod
    def export_image(frame: np.ndarray, file_path: str) -> bool:
        """Save a single annotated frame as image."""
        return cv2.imwrite(file_path, frame)

    @staticmethod
    def export_video(
        frames: list[np.ndarray],
        file_path: str,
        fps: float = 30.0,
    ) -> bool:
        """Write buffered annotated frames to MP4."""
        if not frames:
            return False

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            return False

        for frame in frames:
            writer.write(frame)

        writer.release()
        return True

    @staticmethod
    def export_match_log(rows: list[dict], file_path: str) -> bool:
        """Export match log table to CSV."""
        if not rows:
            return False

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["Time", "Frame", "Similarity", "Person ID", "Status"]
            )
            writer.writeheader()
            writer.writerows(rows)

        return True
