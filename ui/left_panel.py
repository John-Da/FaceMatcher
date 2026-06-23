from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QRadioButton,
    QLineEdit,
    QComboBox,
    QSlider,
)

from core.utils import (
    browse_image_file,
    load_pixmap_into_label,
    list_models,
    list_cameras,
)

class LeftPanel(QWidget):

    viewer_changed = Signal(str)
    reference_face_selected = Signal(str)
    image_selected = Signal(str)
    video_selected = Signal(str)
    webcam_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        group_style = """
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
            }
        """

        # ==================================================
        # Reference Face
        # ==================================================
        reference_group = QGroupBox("Reference Face")
        reference_group.setStyleSheet(group_style)
        reference_layout = QVBoxLayout()
        reference_layout.setSpacing(8)
        reference_layout.setContentsMargins(8, 12, 8, 8)

        self.face_preview = QLabel("No Image Selected")
        self.face_preview.setFixedSize(180, 180)
        self.face_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_preview.setStyleSheet("border: 2px dashed gray;")

        self.upload_face_btn = QPushButton("Upload Face")
        self.upload_face_btn.clicked.connect(self.browse_reference_face)

        reference_layout.addWidget(self.face_preview, alignment=Qt.AlignmentFlag.AlignHCenter)
        reference_layout.addWidget(self.upload_face_btn)
        reference_group.setLayout(reference_layout)

        # ==================================================
        # Source Selection
        # ==================================================
        source_group = QGroupBox("Source Selection")
        source_group.setStyleSheet(group_style)
        source_layout = QVBoxLayout()
        source_layout.setSpacing(8)
        source_layout.setContentsMargins(8, 12, 8, 8)

        self.image_radio = QRadioButton("Image")
        self.video_radio = QRadioButton("Video")
        self.webcam_radio = QRadioButton("Webcam")
        self.video_radio.setChecked(True)

        image_row = QHBoxLayout()
        image_row.setSpacing(6)
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Select image...")
        self.image_browse_btn = QPushButton()
        self.image_browse_btn.setIcon(QIcon("assets/icons/images-regular.png"))
        self.image_browse_btn.setIconSize(QSize(16, 16))
        self.image_browse_btn.setFixedWidth(30)
        image_row.addWidget(self.image_radio)
        image_row.addWidget(self.image_path)
        image_row.addWidget(self.image_browse_btn)

        video_row = QHBoxLayout()
        video_row.setSpacing(6)
        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("Select video...")
        self.video_browse_btn = QPushButton()
        self.video_browse_btn.setIcon(QIcon("assets/icons/folder-open-solid.png"))
        self.video_browse_btn.setIconSize(QSize(16, 16))
        self.video_browse_btn.setFixedWidth(30)
        self.video_browse_btn.clicked.connect(self.browse_video)
        video_row.addWidget(self.video_radio)
        video_row.addWidget(self.video_path)
        video_row.addWidget(self.video_browse_btn)

        webcam_row = QHBoxLayout()
        webcam_row.setSpacing(6)
        self.webcam_combo = QComboBox()
        self.webcam_combo.addItems(list_cameras())
        webcam_row.addWidget(self.webcam_radio)
        webcam_row.addWidget(self.webcam_combo)

        source_layout.addLayout(image_row)
        source_layout.addLayout(video_row)
        source_layout.addLayout(webcam_row)
        source_group.setLayout(source_layout)

        # ==================================================
        # Settings
        # ==================================================
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet(group_style)
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(8)
        settings_layout.setContentsMargins(8, 12, 8, 8)

        threshold_row = QHBoxLayout()
        threshold_row.setSpacing(8)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)
        self.threshold_value = QLabel("0.50")
        self.threshold_value.setFixedWidth(35)
        threshold_row.addWidget(QLabel("Match Threshold"))
        threshold_row.addWidget(self.threshold_slider)
        threshold_row.addWidget(self.threshold_value)

        detector_row = QHBoxLayout()
        detector_row.setSpacing(8)
        self.detector_combo = QComboBox()
        self.detector_combo.addItems(
            list_models("data/models/detectors", ["No models found"])
        )
        detector_row.addWidget(QLabel("Detector"))
        detector_row.addStretch()
        detector_row.addWidget(self.detector_combo)

        reid_row = QHBoxLayout()
        reid_row.setSpacing(8)
        self.reid_combo = QComboBox()
        self.reid_combo.addItems(list_models("data/models/reids", ["No models found"]))
        reid_row.addWidget(QLabel("ReID Model"))
        reid_row.addStretch()
        reid_row.addWidget(self.reid_combo)

        settings_layout.addLayout(threshold_row)
        settings_layout.addLayout(detector_row)
        settings_layout.addLayout(reid_row)
        settings_group.setLayout(settings_layout)

        # ==================================================
        # Controls
        # ==================================================
        controls_group = QGroupBox("Controls")
        controls_group.setStyleSheet(group_style)
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(8)
        controls_layout.setContentsMargins(8, 12, 8, 8)

        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setDisabled(True)
        self.stop_btn = QPushButton("Stop Processing")
        self.export_btn = QPushButton("Export Results")

        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.export_btn)
        controls_group.setLayout(controls_layout)

        # ==================================================
        # Add All Sections
        # ==================================================
        layout.addWidget(reference_group)
        layout.addWidget(source_group)
        layout.addWidget(settings_group)
        layout.addWidget(controls_group)

        # ==================================================
        # Focus Policy
        # ==================================================
        for widget in [
            self.upload_face_btn,
            self.image_browse_btn,
            self.video_browse_btn,
            self.start_btn,
            self.stop_btn,
            self.export_btn,
            self.image_radio,
            self.video_radio,
            self.webcam_radio,
            self.detector_combo,
            self.reid_combo,
            self.webcam_combo,
            self.threshold_slider,
            self.image_path,
            self.video_path,
        ]:
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # ==================================================
        # Signals
        # ==================================================
        self.threshold_slider.valueChanged.connect(self.update_threshold)

        self.image_radio.toggled.connect(
            lambda checked: self.viewer_changed.emit("image") if checked else None
        )
        self.video_radio.toggled.connect(
            lambda checked: self.viewer_changed.emit("video") if checked else None
        )
        self.webcam_radio.toggled.connect(
            lambda checked: (
                (
                    self.viewer_changed.emit("webcam")
                    or self.webcam_selected.emit(self.webcam_combo.currentText())
                )
                if checked
                else None
            )
        )

        self.image_path.textChanged.connect(self.update_start_btn)
        self.video_path.textChanged.connect(self.update_start_btn)
        self.image_radio.toggled.connect(self.update_start_btn)
        self.video_radio.toggled.connect(self.update_start_btn)
        self.webcam_radio.toggled.connect(self.update_start_btn)
        self.webcam_combo.currentIndexChanged.connect(self.on_webcam_changed)

    def update_threshold(self, value):
        self.threshold_value.setText(f"{value / 100:.2f}")

    def update_start_btn(self):
        has_source = (
            self.image_radio.isChecked()
            and bool(self.image_path.text())
            or self.video_radio.isChecked()
            and bool(self.video_path.text())
            or self.webcam_radio.isChecked()
        )
        self.start_btn.setEnabled(has_source)

    def browse_reference_face(self):

        file_path = browse_image_file(self)
        if not file_path:
            return

        load_pixmap_into_label(self.face_preview, file_path)
        self.reference_face_path = file_path
        self.reference_face_selected.emit(file_path)

    def browse_video(self):
        from core.utils import browse_video_file

        file_path = browse_video_file(self)
        if not file_path:
            return

        self.video_path.setText(file_path)
        self.video_radio.setChecked(True)
        self.video_selected.emit(file_path)

    def on_webcam_changed(self, index: int):
        if self.webcam_radio.isChecked():
            self.webcam_selected.emit(self.webcam_combo.itemText(index))

    def _revert_webcam_combo(self, name: str):
        combo = self.left_panel.webcam_combo
        index = combo.findText(name)
        if index >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
