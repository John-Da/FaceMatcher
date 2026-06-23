import sys
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FaceMatcher")
    app.setWindowIcon(QIcon("assets/logos/app_icon.png"))

    window = MainWindow()
    window.setWindowTitle("FaceMatcher")
    window.setWindowIcon(QIcon("assets/logos/app_icon.png"))
    window.resize(1600, 900)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
