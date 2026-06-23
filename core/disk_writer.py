from __future__ import annotations
import cv2
import tempfile
import os
import numpy as np


class DiskWriter:
    """Writes annotated frames to a temp .mp4 file as they arrive."""

    def __init__(self):
        self._writer: cv2.VideoWriter | None = None
        self._path: str | None = None

    def open(self, fps: float, width: int, height: int) -> str:
        """Open a temp file and return its path."""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self._path = tmp.name
        tmp.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._path, fourcc, fps, (width, height))
        return self._path

    def write(self, frame: np.ndarray):
        if self._writer:
            self._writer.write(frame)

    def close(self):
        if self._writer:
            self._writer.release()
            self._writer = None

    def cleanup(self):
        """Delete the temp file when done."""
        self.close()
        if self._path and os.path.exists(self._path):
            os.remove(self._path)
            self._path = None

    @property
    def path(self) -> str | None:
        return self._path
