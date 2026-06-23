import os
import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QStackedWidget,
    QFileDialog,
    QMessageBox,
)

from ui.left_panel import LeftPanel
from ui.video_viewer import VideoViewer
from ui.image_viewer import ImageViewer
from ui.webcam_viewer import WebcamViewer
from ui.match_log import MatchLog
from ui.loading_dialog import LoadingDialog

from core.utils import browse_image_file
from core.processor import Processor
from core.model_loader import ModelLoader
from core.exporter import Exporter
from core.frame_reader import FrameReader

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # ==================================================
        # Left Panel
        # ==================================================
        self.left_panel = LeftPanel()

        # ==================================================
        # Right Side (Viewer + Match Log)
        # ==================================================
        right_layout = QVBoxLayout()

        self.image_viewer = ImageViewer()
        self.video_viewer = VideoViewer()
        self.webcam_viewer = WebcamViewer()

        self.viewer_stack = QStackedWidget()
        self.viewer_stack.addWidget(self.image_viewer)  # index 0
        self.viewer_stack.addWidget(self.video_viewer)  # index 1
        self.viewer_stack.addWidget(self.webcam_viewer)  # index 2

        self.match_log = MatchLog()

        right_layout.addWidget(self.viewer_stack, 4)
        right_layout.addWidget(self.match_log, 1)

        # ==================================================
        # Assemble
        # ==================================================
        main_layout.addWidget(self.left_panel, 1)
        main_layout.addLayout(right_layout, 4)

        # ==================================================
        # Signals
        # ==================================================
        self.left_panel.image_browse_btn.clicked.connect(self._browse_image)
        self.left_panel.viewer_changed.connect(self.show_viewer)
        self.left_panel.viewer_changed.connect(self.on_viewer_changed)
        self.left_panel.video_selected.connect(self.video_viewer.load_video)
        self.left_panel.webcam_selected.connect(self.webcam_viewer.start_camera)

        self.webcam_viewer.start_btn.clicked.connect(self._start_webcam)
        self.webcam_viewer.stop_btn.clicked.connect(self.webcam_viewer.on_stop_btn)
        self.webcam_viewer.revert_camera_signal.connect(self._revert_webcam_combo)

        # set default current_camera from combo without starting
        first_camera = self.left_panel.webcam_combo.currentText()
        if first_camera and first_camera != "No cameras found":
            self.webcam_viewer.current_camera = first_camera

        self.show_viewer("video")

        self.processor = Processor()
        self.reader = None
        self.ref_embedding = None
        self._detector = None
        self._reid = None
        self._loading_dialog = None
        self._model_loader = None

        self.left_panel.start_btn.clicked.connect(self.start_processing)
        self.left_panel.stop_btn.clicked.connect(self.stop_processing)
        self.left_panel.reference_face_selected.connect(self.on_reference_selected)

        self.processor.frame_ready.connect(self.on_frame_ready)
        self.processor.match_found.connect(self.match_log.add_match)
        self.processor.finished.connect(self.on_processing_finished)
        self.processor.error.connect(self.on_processor_error)

        self.left_panel.export_btn.clicked.connect(self.export_results)

    def show_viewer(self, name: str):
        viewers = {
            "image": 0,
            "video": 1,
            "webcam": 2,
        }
        self.viewer_stack.setCurrentIndex(viewers.get(name, 0))

    def _browse_image(self):
        file_path = browse_image_file(self)
        if not file_path:
            return
        self.left_panel.image_path.setText(file_path)
        self.left_panel.image_radio.setChecked(True)
        self.show_viewer("image")
        self.image_viewer.load_image(file_path)

    def on_viewer_changed(self, name: str):
        if name != "webcam":
            self.webcam_viewer.stop_camera(clear_label=True)

    def _start_webcam(self):
        camera_name = self.left_panel.webcam_combo.currentText()
        if camera_name and camera_name != "No cameras found":
            self.webcam_viewer.current_camera = camera_name
            self.webcam_viewer.start_camera(camera_name)

    def _revert_webcam_combo(self, name: str):
        combo = self.left_panel.webcam_combo
        index = combo.findText(name)
        if index >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def _get_selected_models(self) -> tuple[str, str]:
        return (
            self.left_panel.detector_combo.currentText(),
            self.left_panel.reid_combo.currentText(),
        )

    def _models_valid(self, detector_name: str, reid_name: str) -> bool:
        return (
            detector_name not in ("", "No models found")
            and reid_name not in ("", "No models found")
        )

    def _show_loading(self, message: str):
        self._loading_dialog = LoadingDialog(self)
        self._loading_dialog.set_status(message)
        self._loading_dialog.show()

    def _hide_loading(self):
        if self._loading_dialog:
            self._loading_dialog.close()
            self._loading_dialog = None

    def _models_changed(self) -> bool:
        """Check if user switched models since last load."""
        detector_name, reid_name = self._get_selected_models()
        return (
            not hasattr(self, "_loaded_detector_name")
            or self._loaded_detector_name != detector_name
            or self._loaded_reid_name != reid_name
        )

    def _load_models(self, detector_name: str, reid_name: str, on_ready):
        """Load models in background, call on_ready() when done."""
        self._show_loading("Loading models...")
        self.left_panel.start_btn.setEnabled(False)

        self._model_loader = ModelLoader(detector_name, reid_name)
        self._model_loader.status.connect(self._loading_dialog.set_status)
        self._model_loader.finished.connect(
            lambda det, reid: self._on_models_loaded(det, reid, detector_name, reid_name, on_ready)
        )
        self._model_loader.error.connect(self._on_model_load_error)
        self._model_loader.start()

    def _on_models_loaded(self, detector, reid, detector_name, reid_name, on_ready):
        self._detector = detector
        self._reid = reid
        self._loaded_detector_name = detector_name
        self._loaded_reid_name = reid_name
        self._hide_loading()
        on_ready()

    def _on_model_load_error(self, msg: str):
        self._hide_loading()
        self.left_panel.start_btn.setEnabled(True)
        print(f"[Model Load Error] {msg}")

    def on_reference_selected(self, file_path: str):
        detector_name, reid_name = self._get_selected_models()
        if not self._models_valid(detector_name, reid_name):
            return

        if self._models_changed():
            self._load_models(
                detector_name,
                reid_name,
                on_ready=lambda: self._extract_reference(file_path),
            )
        else:
            self._extract_reference(file_path)

    def _extract_reference(self, file_path: str):
        img = cv2.imread(file_path)
        if img is None:
            return

        ref_crop = self._detector.detect_single(img)
        if ref_crop is None:
            ref_crop = img  # fallback: use full image

        self.ref_embedding = self._reid.extract(ref_crop)
        self.left_panel.start_btn.setEnabled(True)

    def start_processing(self):
        if self.ref_embedding is None:
            return

        detector_name, reid_name = self._get_selected_models()
        if not self._models_valid(detector_name, reid_name):
            return

        if self._models_changed():
            # models switched — reload then start
            self._load_models(
                detector_name,
                reid_name,
                on_ready=self._begin_processing,
            )
        else:
            self._begin_processing()

    def _begin_processing(self):
        self.processor._disk_writer.cleanup()
        self._temp_video_path = None

        if self.reader and self.reader.isRunning():
            self.reader.stop()
            self.reader.wait()
        if self.processor.isRunning():
            self.processor.stop()
            self.processor.wait()

        threshold = self.left_panel.threshold_slider.value() / 100.0

        if self.left_panel.image_radio.isChecked():
            source, source_path = "image", self.left_panel.image_path.text()
            self.image_viewer.set_status("Processing...")
        elif self.left_panel.video_radio.isChecked():
            source, source_path = "video", self.left_panel.video_path.text()
        else:
            source, source_path = "webcam", self.left_panel.webcam_combo.currentText()

        is_video = self.left_panel.video_radio.isChecked()

        self.processor.setup(
            detector=self._detector,
            reid=self._reid,
            ref_embedding=self.ref_embedding,
            source=source,
            source_path=source_path,
            threshold=threshold,
            write_to_disk=is_video,   # disk for video, RAM for webcam/image
        )
        self.match_log.clear()
        self.left_panel.start_btn.setEnabled(False)

        if source == "image":
            self.processor.start()
        else:
            self.reader = FrameReader(queue=self.processor.frame_queue)
            self.reader.setup(source=source, source_path=source_path)
            self.reader.fps_ready.connect(self._on_fps_ready)
            self.reader.finished.connect(self.processor.stop)
            self.reader.error.connect(self.on_processor_error)
            self.reader.start()
            self.processor.start()
    
    def closeEvent(self, event):
        self.processor._disk_writer.cleanup()
        super().closeEvent(event)

    def _on_fps_ready(self, fps: float, total_frames: int):
        """FrameReader tells us FPS once the capture opens."""
        self.processor.source_fps = fps
        if self.left_panel.video_radio.isChecked():
            self.video_viewer.enter_processing_mode(fps=fps, total_frames=total_frames)

    def on_processing_finished(self):
        self.left_panel.start_btn.setEnabled(True)

        if self.left_panel.video_radio.isChecked():
            self.video_viewer.set_overlay_frame(None)  # clear overlay
            tmp_path = self.processor.output_video_path
            if tmp_path:
                self.video_viewer.load_video(tmp_path)  # load processed result
                self._temp_video_path = tmp_path
            return

        if self.left_panel.image_radio.isChecked():
            path = self.left_panel.image_path.text()
            if path and self.image_viewer.image_label.text() == "Processing...":
                self.image_viewer.load_image(path)

    def on_frame_ready(self, frame: np.ndarray, frame_num: int):
        if self.left_panel.image_radio.isChecked():
            self.image_viewer.display_frame(frame)
            return

        if self.left_panel.video_radio.isChecked():
            self.video_viewer.set_overlay_frame(frame)  # ← just set overlay
            return

        self._display_on_label(frame, self.webcam_viewer.webcam_label)

    def _display_on_label(self, frame: np.ndarray, label):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def stop_processing(self):
        if self.reader and self.reader.isRunning():
            self.reader.stop()
        self.processor.stop()
        self.video_viewer.stop_processing_mode()
        self.left_panel.start_btn.setEnabled(True)

    def _display_annotated_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

        if self.left_panel.video_radio.isChecked():
            label = self.video_viewer.video_label
        else:
            label = self.webcam_viewer.webcam_label

        pixmap = QPixmap.fromImage(image).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def on_processor_error(self, msg: str):
        print(f"[Processor Error] {msg}")
        self.left_panel.start_btn.setEnabled(True)

    def export_results(self):
        is_video = self.left_panel.video_radio.isChecked()

        if is_video:
            src = getattr(self, "_temp_video_path", None)
            if not src:
                QMessageBox.warning(self, "Export", "No results to export yet.")
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Video", "result.mp4", "Video (*.mp4)"
            )
            if not file_path:
                return

            # Just copy the temp file — it's already encoded
            import shutil
            shutil.copy2(src, file_path)

            csv_path = os.path.splitext(file_path)[0] + "_matches.csv"
            rows = self.match_log.get_rows()
            if rows:
                Exporter.export_match_log(rows, csv_path)
                QMessageBox.information(self, "Export",
                    f"Video saved to:\n{file_path}\n\nMatch log:\n{csv_path}")
            else:
                QMessageBox.information(self, "Export", f"Video saved to:\n{file_path}")
            return

        if self.processor.last_frame is None:
            QMessageBox.warning(self, "Export", "No results to export yet.")
            return

        is_image = self.left_panel.image_radio.isChecked()

        if is_image:
            # --- Export single annotated image ---
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Image",
                "result.png",
                "Images (*.png *.jpg)",
            )
            if not file_path:
                return

            ok = Exporter.export_image(self.processor.last_frame, file_path)
            if ok:
                QMessageBox.information(self, "Export", f"Image saved to:\n{file_path}")
            else:
                QMessageBox.critical(self, "Export", "Failed to save image.")

        else:
            # --- Export annotated video + optional CSV ---
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Video",
                "result.mp4",
                "Video (*.mp4)",
            )
            if not file_path:
                return

            ok = Exporter.export_video(
                self.processor.frame_buffer,
                file_path,
                fps=self.processor.source_fps,
            )

            if ok:
                # also offer CSV export
                csv_path = os.path.splitext(file_path)[0] + "_matches.csv"
                rows = self.match_log.get_rows()
                if rows:
                    Exporter.export_match_log(rows, csv_path)
                    QMessageBox.information(
                        self,
                        "Export",
                        f"Video saved to:\n{file_path}\n\nMatch log saved to:\n{csv_path}",
                    )
                else:
                    QMessageBox.information(
                        self, "Export", f"Video saved to:\n{file_path}"
                    )
            else:
                QMessageBox.critical(self, "Export", "Failed to save video.")
