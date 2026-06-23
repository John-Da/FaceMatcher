from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
)


class LoadingDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Initializing")
        self.setModal(True)
        self.setFixedSize(320, 120)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        self.status_label = QLabel("Loading models...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 13px;")

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate spinner
        self.progress.setTextVisible(False)

        self.dot_label = QLabel("")
        self.dot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dot_label.setStyleSheet("font-size: 11px; color: gray;")

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.dot_label)

        # animated dots
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._animate_dots)
        self._dot_timer.start(400)

    def set_status(self, text: str):
        self.status_label.setText(text)

    def _animate_dots(self):
        self._dot_count = (self._dot_count + 1) % 4
        self.dot_label.setText("●" * self._dot_count + "○" * (3 - self._dot_count))

    def closeEvent(self, event):
        self._dot_timer.stop()
        super().closeEvent(event)
