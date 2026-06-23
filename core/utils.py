from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QLabel

import os
import cv2
import AVFoundation as AVF

def list_models(folder: str, fallback: list[str] = None) -> list[str]:
    if not os.path.exists(folder):
        return fallback or []
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return files if files else (fallback or [])


def list_cameras() -> list[str]:
    # get names from AVFoundation in their order
    device_types = [
        AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera,
        AVF.AVCaptureDeviceTypeExternalUnknown,
    ]
    discovery = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        device_types,
        AVF.AVMediaTypeVideo,
        AVF.AVCaptureDevicePositionUnspecified,
    )
    av_names = [d.localizedName() for d in discovery.devices()]

    # suppress opencv stderr
    devnull = open(os.devnull, "w")
    old_stderr = os.dup(2)
    os.dup2(devnull.fileno(), 2)

    # find which opencv indices are valid
    valid_indices = []
    try:
        for i in range(len(av_names)):
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                valid_indices.append(i)
                cap.release()
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        devnull.close()

    # pair valid opencv indices with av names in reverse order
    av_names_reversed = list(reversed(av_names))
    result = []
    for i, idx in enumerate(valid_indices):
        name = av_names_reversed[i] if i < len(av_names_reversed) else f"Camera {idx}"
        result.append(f"{idx}: {name}")

    return result or ["No cameras found"]


def browse_image_file(parent) -> str | None:
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Select Image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
    )
    return file_path or None


def load_pixmap_into_label(label: QLabel, file_path: str) -> QPixmap:
    pixmap = QPixmap(file_path)
    pixmap = pixmap.scaled(
        label.size(),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    label.setPixmap(pixmap)
    label.setText("")
    return pixmap


def browse_video_file(parent) -> str | None:
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Select Video",
        "",
        "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)",
    )
    return file_path or None
