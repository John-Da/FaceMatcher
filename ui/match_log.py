import time
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLabel,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor


class MatchLog(QWidget):

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Match Log")
        header.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px 0;")

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setMinimumSize(800, 200)

        self._min_update_interval = 0.15  # seconds, ~6-7 UI updates/sec max

        self.table.setHorizontalHeaderLabels(
            [
                "Time",
                "Frame",
                "Similarity",
                "Person ID",
                "Status",
            ]
        )

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(header)
        layout.addWidget(self.table)

    # def add_match(
    #     self, timestamp, frame_num, similarity, person_id, status="Match Found"
    # ):
    #     row = self.table.rowCount()
    #     self.table.insertRow(row)

    #     items = [
    #         str(timestamp),
    #         str(frame_num),
    #         f"{similarity:.0%}",
    #         str(person_id),
    #         status,
    #     ]

    #     for col, text in enumerate(items):
    #         item = QTableWidgetItem(text)
    #         item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    #         # color similarity cell by score
    #         if col == 2:
    #             if similarity >= 0.85:
    #                 item.setForeground(QColor("#22c55e"))  # green
    #             elif similarity >= 0.7:
    #                 item.setForeground(QColor("#f59e0b"))  # amber
    #             else:
    #                 item.setForeground(QColor("#94a3b8"))  # gray

    #         self.table.setItem(row, col, item)

    #     # auto-scroll to latest row
    #     self.table.scrollToBottom()

    # def clear(self):
    #     self.table.setRowCount(0)

    def add_match(
        self, timestamp, frame_num, similarity, person_id, status="Match Found"
    ):
        same_person = person_id == self._last_person_id and self._last_row >= 0
        now = time.monotonic()

        if same_person:
            row = self._last_row
            # throttle: skip the UI write if we just updated this row
            if now - self._last_update_ts < self._min_update_interval:
                return
        else:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._last_row = row
            self._last_person_id = person_id

        self._last_update_ts = now

        items = [
            str(timestamp),
            str(frame_num),
            f"{similarity:.0%}",
            str(person_id),
            status,
        ]
        for col, text in enumerate(items):
            item = self.table.item(row, col)
            if item is None:
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)
            item.setText(text)
            if col == 2:
                if similarity >= 0.85:
                    item.setForeground(QColor("#22c55e"))
                elif similarity >= 0.7:
                    item.setForeground(QColor("#f59e0b"))
                else:
                    item.setForeground(QColor("#94a3b8"))

        self.table.scrollToBottom()

    def clear(self):
        self.table.setRowCount(0)
        self._last_person_id = None
        self._last_row = -1

    def get_rows(self) -> list[dict]:
        """Return all rows as list of dicts for CSV export."""
        rows = []
        for row in range(self.table.rowCount()):
            rows.append({
                "Time":      self.table.item(row, 0).text(),
                "Frame":     self.table.item(row, 1).text(),
                "Similarity": self.table.item(row, 2).text(),
                "Person ID": self.table.item(row, 3).text(),
                "Status":    self.table.item(row, 4).text(),
            })
        return rows
