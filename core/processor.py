from __future__ import annotations

import time
import cv2
import numpy as np
from queue import Queue, Empty
from PySide6.QtCore import QThread, Signal

from core.detector import Detector
from core.reid import ReID
from core.disk_writer import DiskWriter

# Tracker is optional — graceful fallback to detection-only if boxmot missing
try:
    from core.tracker import Tracker

    _TRACKER_AVAILABLE = True
except ImportError:
    _TRACKER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_track(
    frame: np.ndarray,
    box: tuple,
    track_id: int,
    similarity: float,
    is_match: bool,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    color = (0, 255, 0) if is_match else (200, 200, 200)
    thickness = 2 if is_match else 1

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if is_match:
        label = f"ID {track_id}  {similarity:.0%}"
    else:
        label = f"ID {track_id}"

    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - baseline - 4),
        (x1 + text_w, y1),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0) if is_match else (30, 30, 30),
        2,
    )
    return frame


# ---------------------------------------------------------------------------
# Processor thread
# ---------------------------------------------------------------------------


class Processor(QThread):

    # annotated frame + frame_num  (emitted for ALL sources including video)
    frame_ready = Signal(np.ndarray, int)
    # (timestamp_str, frame_num, similarity, track_id)
    match_found = Signal(str, int, float, int)
    finished = Signal()
    error = Signal(str)
    fps_ready = Signal(float)

    def __init__(self):
        super().__init__()
        self.detector: Detector | None = None
        self.reid: ReID | None = None
        self.ref_embedding: np.ndarray | None = None

        self.source: str | None = None
        self.source_path: str | None = None
        self.threshold: float = 0.5

        self._running = False
        self.last_frame: np.ndarray | None = None
        self.frame_buffer: list[np.ndarray] = []
        self.source_fps: float = 30.0
        self.output_video_path: str | None = None
        self.write_to_disk: bool = False

        self.frame_queue: Queue = Queue(maxsize=2)
        self._tracker: Tracker | None = None
        self._disk_writer = DiskWriter()

        # Tracks the last person_id that was actually logged to the match log,
        # so we only log on a CHANGE of identity, not every frame it persists.
        self._last_logged_id: int | None = None

    def setup(
        self,
        detector,
        reid,
        ref_embedding,
        source,
        source_path,
        threshold,
        write_to_disk: bool = False,
    ):
        self.detector = detector
        self.reid = reid
        self.ref_embedding = ref_embedding
        self.source = source
        self.source_path = source_path
        self.threshold = threshold
        self.write_to_disk = write_to_disk
        self.last_frame = None
        self.frame_buffer = []
        self.output_video_path = None
        self._last_logged_id = None  # reset on every new run

    def run(self):
        self._running = True

        # Build a fresh tracker for this run
        if _TRACKER_AVAILABLE:
            self._tracker = Tracker(fps=self.source_fps)
        else:
            self._tracker = None
            # print(
            #     "[Processor] boxmot not found — running without StrongSORT. "
            #     "Install with:  pip install boxmot"
            # )

        if self.source == "image":
            self._process_image()
        elif self.source in ("video", "webcam"):
            self._process_stream()

        self.finished.emit()

    def stop(self):
        self._running = False
        try:
            self.frame_queue.put_nowait(None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------
    def _process_image(self):
        frame = cv2.imread(self.source_path)
        if frame is None:
            self.error.emit(f"Cannot read image: {self.source_path}")
            return

        annotated, matches = self._process_frame(frame, 0)
        self.last_frame = annotated.copy()
        self.frame_ready.emit(annotated, 0)
        self._log_new_matches("--", 0, matches)

    # ------------------------------------------------------------------
    # Video / Webcam stream
    # ------------------------------------------------------------------
    def _process_stream(self):
        frame_num = 0
        writer_opened = False
        fps_window_start = time.time()
        fps_window_count = 0

        while self._running:
            item = self.frame_queue.get()
            if item is None:
                break

            frame = item
            annotated, matches = self._process_frame(frame, frame_num)
            self.last_frame = annotated.copy()

            if self.write_to_disk and not writer_opened:
                h, w = annotated.shape[:2]
                self._disk_writer.open(self.source_fps, w, h)
                writer_opened = True

            if self.write_to_disk:
                self._disk_writer.write(annotated)
            else:
                self.frame_buffer.append(annotated.copy())

            self.frame_ready.emit(annotated, frame_num)

            # Live fps — measured every 1 second, based on frames actually emitted
            fps_window_count += 1
            now = time.time()
            if now - fps_window_start >= 1.0:
                live_fps = fps_window_count / (now - fps_window_start)
                self.fps_ready.emit(live_fps)
                fps_window_count = 0
                fps_window_start = now

            current_sec = frame_num / self.source_fps
            timestamp = self._format_time(current_sec)
            self._log_new_matches(timestamp, frame_num, matches)

            frame_num += 1

        if self.write_to_disk:
            self._disk_writer.close()
            self.output_video_path = self._disk_writer.path

    # ------------------------------------------------------------------
    # Match log filtering — only emit on identity CHANGE
    # ------------------------------------------------------------------
    # def _log_new_matches(
    #     self, timestamp: str, frame_num: int, matches: list[tuple[int, float]]
    # ):
    #     """
    #     Given this frame's matches, emit match_found ONLY for the first match
    #     whose track_id differs from the last one we logged. This turns a
    #     per-frame stream of "ID 1, ID 1, ID 1, ID 10, ID 10, ID 2, ..." into
    #     a clean transition log: "ID 1, ID 10, ID 2".
    #     """
    #     if not matches:
    #         return

    #     # If multiple matches exist in one frame, take the first — this keeps
    #     # "current tracked identity" as a single value, consistent frame to frame.
    #     track_id, similarity = matches[0]

    #     if track_id == self._last_logged_id:
    #         return  # same person as last logged entry — skip, no new row

    #     self._last_logged_id = track_id
    #     self.match_found.emit(timestamp, frame_num, similarity, track_id)

    def _log_new_matches(
        self, timestamp: str, frame_num: int, matches: list[tuple[int, float]]
    ):
        # Emit for every currently-matched person this frame — MatchLog
        # decides per-ID whether that's a new row or an update to an
        # existing one.
        for track_id, similarity in matches:
            self.match_found.emit(timestamp, frame_num, similarity, track_id)

    # ------------------------------------------------------------------
    # Per-frame logic
    # ------------------------------------------------------------------
    def _process_frame(
        self, frame: np.ndarray, frame_num: int
    ) -> tuple[np.ndarray, list[tuple[int, float]]]:
        """
        Returns (annotated_frame, [(track_id, similarity), ...])
        """
        annotated = frame.copy()
        matches: list[tuple[int, float]] = []

        t0 = time.perf_counter()
        detections = self.detector.detect(
            frame,
            imgsz=640 if self.source == "webcam" else None,
        )
        t1 = time.perf_counter()

        embeddings: list[np.ndarray] = self.reid.extract_batch(
            [det["crop"] for det in detections]
        )
        t2 = time.perf_counter()

        # --- Tracking path (StrongSORT) ---
        if self._tracker is not None and detections:
            tracks = self._tracker.update(detections, embeddings, frame)
            t3 = time.perf_counter()

            # Map track_id → best embedding via IoU between track box and det box
            # (StrongSORT already matched them; we just need the similarity)
            track_similarities: dict[int, float] = {}
            for track in tracks:
                best_sim = 0.0
                tx1, ty1, tx2, ty2 = track["box"]
                best_iou = 0.0
                best_idx = 0
                for i, det in enumerate(detections):
                    iou = _iou(track["box"], det["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if embeddings:
                    best_sim = float(
                        self.reid.similarity(self.ref_embedding, embeddings[best_idx])
                    )
                track_similarities[track["track_id"]] = best_sim

            # Draw all tracks; highlight matches
            for track in tracks:
                tid = track["track_id"]
                sim = track_similarities.get(tid, 0.0)
                is_match = sim >= self.threshold
                _draw_track(annotated, track["box"], tid, sim, is_match)
                if is_match:
                    matches.append((tid, sim))

        # --- Fallback: no tracker (detection-only) ---
        elif not self._tracker:
            t3 = time.perf_counter()
            for person_id, (det, emb) in enumerate(zip(detections, embeddings)):
                similarity = float(self.reid.similarity(self.ref_embedding, emb))
                is_match = similarity >= self.threshold
                _draw_track(annotated, det["box"], person_id, similarity, is_match)
                if is_match:
                    matches.append((person_id, similarity))
        else:
            t3 = time.perf_counter()

        t4 = time.perf_counter()

        # if frame_num % 30 == 0:
        #     print(
        #         f"[frame {frame_num}] n_dets={len(detections)}  "
        #         f"detect={1000*(t1-t0):.1f}ms  "
        #         f"reid={1000*(t2-t1):.1f}ms  "
        #         f"track/draw={1000*(t4-t2):.1f}ms  "
        #         f"TOTAL={1000*(t4-t0):.1f}ms"
        #     )

        return annotated, matches

    # ------------------------------------------------------------------
    # Util
    # ------------------------------------------------------------------
    def _format_time(self, seconds: float) -> str:
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"


def _iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)
