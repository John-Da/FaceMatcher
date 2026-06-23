from __future__ import annotations
import numpy as np

try:
    from boxmot.trackers import OccluBoost
except (ImportError, Exception) as e:
    raise ImportError(f"boxmot OccluBoost unavailable: {e}")


class Tracker:

    def __init__(self, fps: float = 30.0, **kwargs):
        self.fps = fps
        self._tracker = OccluBoost(
            det_thresh=0.4,  # raise if getting too many false detections
            max_age=50,  # how many frames to keep a lost track alive
            min_hits=1,  # lower = tracks appear faster (1 = instant)
            iou_threshold=0.3,
        )

    def update(
        self,
        detections: list[dict],
        embeddings: list[np.ndarray],
        frame: np.ndarray,
    ) -> list[dict]:

        if not detections:
            empty = np.empty((0, 6), dtype=np.float32)
            self._tracker.update(empty, frame)
            return []

        # boxmot expects: (N, 6) → [x1, y1, x2, y2, conf, cls]
        dets_array = np.array(
            [[*d["box"], d.get("score", 1.0), 0] for d in detections],
            dtype=np.float32,
        )

        # tracks: (M, 8) → [x1, y1, x2, y2, id, conf, cls, det_ind]
        tracks_raw = self._tracker.update(dets_array, frame)

        results = []
        if tracks_raw is not None and len(tracks_raw):
            for t in tracks_raw:
                x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                track_id = int(t[4])
                conf = float(t[5])
                results.append(
                    {
                        "track_id": track_id,
                        "box": (x1, y1, x2, y2),
                        "confidence": conf,
                    }
                )

        return results
