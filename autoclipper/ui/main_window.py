"""
AutoClipper – redesigned dark desktop UI.

Premium dark tool aesthetic: near-black background, deep-red accent,
tabbed layout, animated progress, colored terminal log.
"""

from __future__ import annotations

import html as _html
import logging
from pathlib import Path

from PyQt5.QtCore import (
    Q_ARG,
    QEasingCurve,
    QMetaObject,
    QPropertyAnimation,
    QAbstractAnimation,
    Qt,
    QThread,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

STYLESHEET = """
/* ── Base ─────────────────────────────────────────────────── */
QWidget {
    background-color: #0a0a0a;
    color: #f0f0f0;
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    outline: none;
}

/* ── Tab widget ───────────────────────────────────────────── */
QTabWidget::pane {
    border: none;
    background: #0a0a0a;
    top: -1px;
}
QTabWidget::tab-bar {
    alignment: left;
}
QTabBar {
    background: #0a0a0a;
    border-bottom: 1px solid #1a1a1a;
}
QTabBar::tab {
    background: transparent;
    color: #888888;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 10px 28px;
    font-size: 13px;
    min-width: 90px;
}
QTabBar::tab:selected {
    color: #f0f0f0;
    border-bottom: 2px solid #c0392b;
}
QTabBar::tab:hover:!selected {
    color: #cccccc;
    border-bottom: 2px solid #3a3a3a;
}
QTabBar::tab:!selected {
    margin-top: 2px;
}

/* ── Scroll areas ─────────────────────────────────────────── */
QScrollArea {
    border: none;
    background: #0a0a0a;
}
QScrollBar:vertical {
    background: #0a0a0a;
    width: 5px;
    margin: 0;
    border: none;
}
QScrollBar::handle:vertical {
    background: #2a2a2a;
    border-radius: 2px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #3d3d3d;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    height: 0; width: 0; border: none; background: none;
}

/* ── Line edits ───────────────────────────────────────────── */
QLineEdit {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    color: #f0f0f0;
    padding: 10px 12px;
    font-size: 14px;
    selection-background-color: #c0392b;
    selection-color: #ffffff;
}
QLineEdit:hover { border-color: #444444; }
QLineEdit:focus { border-color: #c0392b; background: #1c1010; }

/* ── Spin boxes ───────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    color: #f0f0f0;
    padding: 9px 12px;
    font-size: 14px;
}
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #444444; }
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #c0392b; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: #222222;
    border: none;
    border-left: 1px solid #2a2a2a;
    width: 22px;
    border-radius: 0;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background: #333333;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #888888;
    width: 0; height: 0;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #888888;
    width: 0; height: 0;
}

/* ── Buttons ──────────────────────────────────────────────── */
QPushButton {
    background: #161616;
    color: #888888;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
}
QPushButton:hover {
    background: #1e1e1e;
    color: #cccccc;
    border-color: #444444;
}
QPushButton:pressed { background: #111111; }
QPushButton:disabled { color: #3a3a3a; border-color: #1e1e1e; }

QPushButton#primary {
    background: #c0392b;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 14px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.4px;
}
QPushButton#primary:hover { background: #e74c3c; }
QPushButton#primary:pressed {
    background: #a93226;
    padding-top: 15px;
    padding-bottom: 13px;
}
QPushButton#primary:disabled {
    background: #3a1010;
    color: #664444;
}

/* ── Sliders ──────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px;
    background: #2a2a2a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #c0392b;
    width: 16px;
    height: 16px;
    border-radius: 8px;
    margin: -6px 0;
    border: 2px solid #0a0a0a;
}
QSlider::handle:horizontal:hover { background: #e74c3c; }
QSlider::sub-page:horizontal {
    background: #c0392b;
    border-radius: 2px;
}

/* ── Progress bar ─────────────────────────────────────────── */
QProgressBar {
    background: #161616;
    border: none;
    border-radius: 4px;
    max-height: 8px;
    min-height: 8px;
    font-size: 0px;
}
QProgressBar::chunk {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #c0392b, stop:1 #e74c3c
    );
    border-radius: 4px;
}

/* ── Log terminal ─────────────────────────────────────────── */
QTextEdit#log {
    background: #050505;
    border: 1px solid #1a1a1a;
    border-radius: 8px;
    font-family: "JetBrains Mono", "Consolas", "Monaco", "Courier New", monospace;
    font-size: 12px;
    color: #888888;
    padding: 6px 8px;
}

/* ── Collapsible toggle ───────────────────────────────────── */
QToolButton#collapse_toggle {
    background: transparent;
    color: #666666;
    border: none;
    font-size: 12px;
    text-align: left;
    padding: 2px 0;
}
QToolButton#collapse_toggle:hover { color: #aaaaaa; }

/* ── Message boxes ────────────────────────────────────────── */
QMessageBox { background: #161616; }
QMessageBox QLabel { color: #f0f0f0; }
QMessageBox QPushButton { min-width: 80px; }

/* ── Multi-path list ──────────────────────────────────────── */
QListWidget {
    background: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    outline: none;
    padding: 3px;
}
QListWidget::item {
    background: transparent;
    border-radius: 4px;
    padding: 0;
    margin: 1px 0;
}
QListWidget::item:hover { background: #181818; }
QListWidget::item:selected { background: #1a1010; }

QPushButton#add_btn {
    background: transparent;
    color: #555555;
    border: 1px dashed #2a2a2a;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
}
QPushButton#add_btn:hover {
    color: #c0392b;
    border-color: #c0392b;
    border-style: solid;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Custom widgets
# ─────────────────────────────────────────────────────────────────────────────

class LogView(QTextEdit):
    """Terminal-style log panel with per-level coloring."""

    _COLORS = {
        logging.CRITICAL: "#ff4444",
        logging.ERROR:    "#ff4444",
        logging.WARNING:  "#ffaa33",
        logging.INFO:     "#888888",
        logging.DEBUG:    "#444444",
    }
    _SUCCESS_KW = ("done.", "complete", "saved", "extracted", "100%", "best model")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setObjectName("log")
        self.document().setMaximumBlockCount(3000)
        # Remove default paragraph spacing so lines are compact
        self.document().setDefaultStyleSheet(
            "p, pre { margin: 0; padding: 0; line-height: 1.4; }"
        )

    @pyqtSlot(str)
    def append_html(self, snippet: str) -> None:
        cursor = self.textCursor()
        cursor.movePosition(cursor.End)
        self.setTextCursor(cursor)
        self.insertHtml(snippet)
        self.ensureCursorVisible()

    def append_record(self, record: logging.LogRecord, formatted: str) -> None:
        """Thread-safe: dispatch colored HTML to the main thread."""
        color = self._COLORS.get(record.levelno, "#888888")
        if record.levelno == logging.INFO:
            low = formatted.lower()
            if any(kw in low for kw in self._SUCCESS_KW):
                color = "#00cc66"

        escaped = _html.escape(formatted)
        snippet = (
            f'<span style="color:{color};white-space:pre;font-family:monospace">'
            f'{escaped}</span><br>'
        )
        QMetaObject.invokeMethod(
            self, "append_html",
            Qt.QueuedConnection,
            Q_ARG(str, snippet),
        )


class _ColorLogHandler(logging.Handler):
    def __init__(self, view: LogView):
        super().__init__()
        self._view = view

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._view.append_record(record, msg)
        except Exception:
            self.handleError(record)


class CollapsibleSection(QWidget):
    """A toggle-able section with animated expand/collapse."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title

        self.toggle = QToolButton(self)
        self.toggle.setObjectName("collapse_toggle")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.setText(f"▶  {title}")
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle.clicked.connect(self._on_toggle)

        self.content = QWidget()
        self.content.setMaximumHeight(0)
        self.content.setMinimumHeight(0)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 8, 0, 0)
        self.content_layout.setSpacing(16)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.toggle)
        root.addWidget(self.content)

        self._anim = QPropertyAnimation(self.content, b"maximumHeight")
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.InOutQuart)

    def _on_toggle(self, checked: bool) -> None:
        arrow = "▼" if checked else "▶"
        self.toggle.setText(f"{arrow}  {self._title}")
        if checked:
            self.content.setMaximumHeight(16777215)
            target = max(
                self.content.sizeHint().height(),
                self.content_layout.sizeHint().height() + 20,
            )
            self.content.setMaximumHeight(0)
            self._anim.setStartValue(0)
            self._anim.setEndValue(target)
        else:
            self._anim.setStartValue(self.content.maximumHeight())
            self._anim.setEndValue(0)
        self._anim.start()

    def add_widget(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)


class LabeledSlider(QWidget):
    """Horizontal slider with a live value label."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        value: float,
        scale: int = 100,
        decimals: int = 2,
        parent=None,
    ):
        super().__init__(parent)
        self._scale = scale
        self._decimals = decimals

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(minimum * scale), int(maximum * scale))
        self.slider.setValue(int(value * scale))

        self.val_label = QLabel(f"{value:.{decimals}f}")
        self.val_label.setFixedWidth(46)
        self.val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val_label.setStyleSheet("color: #c0392b; font-size: 13px; font-weight: 600;")

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        h.addWidget(self.slider)
        h.addWidget(self.val_label)

        self.slider.valueChanged.connect(self._on_changed)

    def _on_changed(self, v: int) -> None:
        fval = v / self._scale
        self.val_label.setText(f"{fval:.{self._decimals}f}")
        self.valueChanged.emit(fval)

    def value(self) -> float:
        return self.slider.value() / self._scale

    def setValue(self, v: float) -> None:
        self.slider.setValue(int(v * self._scale))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-path list widget
# ─────────────────────────────────────────────────────────────────────────────

_VIDEO_FILTER = "Video files (*.mp4 *.mov *.m4v *.mkv *.avi);;All files (*)"


class _PathRow(QWidget):
    """Single row inside MultiPathList – shows path name + remove button."""

    remove_requested = pyqtSignal()

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = path
        self.setToolTip(str(path))

        h = QHBoxLayout(self)
        h.setContentsMargins(10, 4, 6, 4)
        h.setSpacing(8)

        kind = QLabel("FILE" if path.is_file() else "DIR")
        kind.setFixedWidth(30)
        kind.setStyleSheet(
            "color: #c0392b; font-size: 9px; font-weight: 700; letter-spacing: 0.5px;"
        )

        name_lbl = QLabel(path.name or str(path))
        name_lbl.setStyleSheet("color: #cccccc; font-size: 12px;")
        name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        rm = QPushButton("×")
        rm.setFixedSize(22, 22)
        rm.setStyleSheet(
            "QPushButton { background: transparent; color: #555555; border: none; "
            "font-size: 15px; font-weight: 300; padding: 0; }"
            "QPushButton:hover { color: #ff4444; }"
        )
        rm.setCursor(Qt.PointingHandCursor)
        rm.clicked.connect(self.remove_requested)

        h.addWidget(kind)
        h.addWidget(name_lbl)
        h.addWidget(rm)


class MultiPathList(QWidget):
    """
    An add/remove list for multiple video file and/or folder paths.

    Shows a QListWidget with per-item remove buttons, plus "Add File"
    and/or "Add Folder" buttons underneath.
    """

    changed = pyqtSignal()

    def __init__(
        self,
        accept_files: bool = True,
        accept_dirs: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._accept_files = accept_files
        self._accept_dirs = accept_dirs
        self._paths: list[Path] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.NoSelection)
        self._list.setFocusPolicy(Qt.NoFocus)
        self._list.setSpacing(1)
        self._list.setMinimumHeight(76)
        self._list.setMaximumHeight(200)
        root.addWidget(self._list)

        btn_row = QWidget()
        btn_h = QHBoxLayout(btn_row)
        btn_h.setContentsMargins(0, 0, 0, 0)
        btn_h.setSpacing(6)

        if accept_files:
            b = QPushButton("+ Add File")
            b.setObjectName("add_btn")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(self._browse_files)
            btn_h.addWidget(b)

        if accept_dirs:
            b = QPushButton("+ Add Folder")
            b.setObjectName("add_btn")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(self._browse_dir)
            btn_h.addWidget(b)

        btn_h.addStretch()
        root.addWidget(btn_row)

    # ── browsing ──────────────────────────────────────────────────────────────

    def _browse_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self.window(), "Select video files", "", _VIDEO_FILTER
        )
        for p in paths:
            self._add(Path(p))

    def _browse_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(self.window(), "Select folder")
        if p:
            self._add(Path(p))

    # ── internal list management ──────────────────────────────────────────────

    def _add(self, path: Path) -> None:
        if path in self._paths:
            return
        self._paths.append(path)

        row_w = _PathRow(path)
        item = QListWidgetItem(self._list)
        item.setSizeHint(row_w.sizeHint())
        self._list.addItem(item)
        self._list.setItemWidget(item, row_w)

        # capture both in the lambda to avoid closure-over-loop issues
        row_w.remove_requested.connect(
            lambda _p=path, _item=item: self._remove(_p, _item)
        )
        self.changed.emit()

    def _remove(self, path: Path, item: QListWidgetItem) -> None:
        row = self._list.row(item)
        if row >= 0:
            self._list.takeItem(row)
        if path in self._paths:
            self._paths.remove(path)
        self.changed.emit()

    # ── public API ────────────────────────────────────────────────────────────

    def paths(self) -> list[Path]:
        return list(self._paths)

    def is_empty(self) -> bool:
        return not self._paths


# ─────────────────────────────────────────────────────────────────────────────
# Worker threads
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionWorker(QThread):
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, base_config, input_paths: list[Path]):
        super().__init__()
        self.base_config = base_config
        self.input_paths = input_paths

    def run(self):
        import copy
        from autoclipper.pipeline import run

        all_results: list = []
        total = len(self.input_paths)

        for i, path in enumerate(self.input_paths):
            cfg = copy.copy(self.base_config)
            cfg.input_video = path

            def cb(msg: str, pct: float, _i: int = i, _n: int = total) -> None:
                overall = (_i + pct) / _n
                self.progress.emit(f"[{_i + 1}/{_n}]  {msg}", overall)

            try:
                results = run(cfg, progress_cb=cb)
                all_results.extend(results)
            except Exception as e:
                log.exception("Extraction failed on %s", path.name)
                self.error.emit(f"{path.name}: {e}")

        self.finished.emit(all_results)


class TrainingWorker(QThread):
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        positives_dirs: list[Path],
        source_paths: list[Path],
        output_model: Path,
        epochs: int,
        batch_size: int,
    ):
        super().__init__()
        self.positives_dirs = positives_dirs
        self.source_paths = source_paths
        self.output_model = output_model
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        try:
            from autoclipper.trainer import build_and_train
            build_and_train(
                self.positives_dirs,
                self.source_paths,
                self.output_model,
                epochs=self.epochs,
                batch_size=self.batch_size,
                progress_cb=lambda m, p: self.progress.emit(m, p),
            )
            self.finished.emit(str(self.output_model))
        except Exception as e:
            log.exception("Training failed")
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# UI building helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "color: #f0f0f0; font-size: 15px; font-weight: 600; padding-top: 4px;"
    )
    return lbl


def _muted_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #888888; font-size: 12px;")
    return lbl


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet("background: #1e1e1e; border: none;")
    return line


def _browse_btn(label: str = "…") -> QPushButton:
    btn = QPushButton(label)
    btn.setFixedWidth(52)
    btn.setFixedHeight(40)
    btn.setCursor(Qt.PointingHandCursor)
    return btn


def _field(
    label_text: str,
    widget: QWidget,
    browse_btn: QPushButton | None = None,
) -> QWidget:
    """Label above + [input  browse?] row."""
    container = QWidget()
    vbox = QVBoxLayout(container)
    vbox.setContentsMargins(0, 0, 0, 0)
    vbox.setSpacing(5)
    vbox.addWidget(_muted_label(label_text))

    if browse_btn is not None:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(widget)
        h.addWidget(browse_btn)
        vbox.addWidget(row)
    else:
        vbox.addWidget(widget)

    return container


def _spin_row(label_text: str, spin: QWidget) -> QWidget:
    """Label on the left, spin on the right in a fixed-width column."""
    container = QWidget()
    vbox = QVBoxLayout(container)
    vbox.setContentsMargins(0, 0, 0, 0)
    vbox.setSpacing(5)
    vbox.addWidget(_muted_label(label_text))
    spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    vbox.addWidget(spin)
    return container


def _scroll_tab(content_widget: QWidget) -> QScrollArea:
    """Wrap content_widget in a frameless, horizontally-fixed scroll area."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(content_widget)
    return scroll


def _make_line_edit(placeholder: str = "", text: str = "") -> QLineEdit:
    le = QLineEdit()
    if placeholder:
        le.setPlaceholderText(placeholder)
    if text:
        le.setText(text)
    return le


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoClipper")
        self.setMinimumSize(700, 800)
        self._worker: QThread | None = None
        self._prog_anim = None

        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._setup_logging()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        root_vbox = QVBoxLayout(root)
        root_vbox.setContentsMargins(0, 0, 0, 0)
        root_vbox.setSpacing(0)

        root_vbox.addWidget(self._build_header())
        root_vbox.addWidget(self._build_tabs(), stretch=1)
        root_vbox.addWidget(self._build_bottom_panel())

        # Animated progress bar value
        self._prog_anim = QPropertyAnimation(self.progress_bar, b"value")
        self._prog_anim.setDuration(220)
        self._prog_anim.setEasingCurve(QEasingCurve.OutCubic)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(46)
        header.setStyleSheet("background: #0d0d0d; border-bottom: 1px solid #1a1a1a;")

        h = QHBoxLayout(header)
        h.setContentsMargins(20, 0, 20, 0)

        dot = QLabel("●")
        dot.setStyleSheet("color: #c0392b; font-size: 11px;")
        dot.setFixedWidth(18)

        title = QLabel("AutoClipper")
        title.setStyleSheet(
            "color: #f0f0f0; font-size: 13px; font-weight: 600; letter-spacing: 1px;"
        )

        h.addWidget(dot)
        h.addWidget(title)
        h.addStretch()
        return header

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(_scroll_tab(self._build_extract_tab()), "Extract")
        self.tabs.addTab(_scroll_tab(self._build_train_tab()), "Train")
        self.tabs.currentChanged.connect(self._animate_tab_in)
        return self.tabs

    def _build_extract_tab(self) -> QWidget:
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(20, 24, 20, 24)
        vbox.setSpacing(16)

        # ── Source ────────────────────────────────────────────────────────────
        vbox.addWidget(_section_header("Source"))

        self.ml_input = MultiPathList(accept_files=True, accept_dirs=True)
        vbox.addWidget(_field(
            "Input videos / season folders  (add as many as you like)",
            self.ml_input,
        ))

        self.le_character = _make_line_edit("e.g.  Walter White   (leave blank to skip)")
        vbox.addWidget(_field("Character name", self.le_character))

        self.le_refs = _make_line_edit("5 – 10 reference face photos for a new character")
        btn_refs = _browse_btn()
        btn_refs.clicked.connect(self._browse_refs)
        vbox.addWidget(_field("Reference images  (new character)", self.le_refs, btn_refs))

        self.le_library = _make_line_edit("Existing clip folder to learn face from")
        btn_lib = _browse_btn()
        btn_lib.clicked.connect(self._browse_library)
        vbox.addWidget(_field("Clips library  (known character)", self.le_library, btn_lib))

        vbox.addWidget(_divider())

        # ── Output ────────────────────────────────────────────────────────────
        vbox.addWidget(_section_header("Output"))

        self.le_output = _make_line_edit(text="output_clips")
        btn_out = _browse_btn()
        btn_out.clicked.connect(self._browse_output)
        vbox.addWidget(_field("Output folder", self.le_output, btn_out))

        self.le_model = _make_line_edit("Optional: trained aesthetic model  (.pt)")
        btn_model = _browse_btn()
        btn_model.clicked.connect(self._browse_model)
        vbox.addWidget(_field("Aesthetic model", self.le_model, btn_model))

        vbox.addWidget(_divider())

        # ── Settings (collapsible) ────────────────────────────────────────────
        settings = CollapsibleSection("Settings")
        vbox.addWidget(settings)

        self.sld_motion = LabeledSlider(0.1, 20.0, 1.2, scale=10, decimals=1)
        settings.add_widget(_field("Min motion score", self.sld_motion))

        self.sld_aesthetic = LabeledSlider(0.1, 1.0, 0.55, scale=100, decimals=2)
        settings.add_widget(_field("Min aesthetic score", self.sld_aesthetic))

        self.sld_scene = LabeledSlider(5.0, 100.0, 27.0, scale=10, decimals=1)
        settings.add_widget(_field("Scene cut threshold", self.sld_scene))

        vbox.addStretch()

        # ── Action button ─────────────────────────────────────────────────────
        self.btn_extract = QPushButton("Extract Clips")
        self.btn_extract.setObjectName("primary")
        self.btn_extract.setCursor(Qt.PointingHandCursor)
        self.btn_extract.clicked.connect(self._start_extraction)
        vbox.addWidget(self.btn_extract)

        return page

    def _build_train_tab(self) -> QWidget:
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(20, 24, 20, 24)
        vbox.setSpacing(16)

        # ── Training data ─────────────────────────────────────────────────────
        vbox.addWidget(_section_header("Training Data"))

        self.ml_train_positives = MultiPathList(accept_files=False, accept_dirs=True)
        vbox.addWidget(_field(
            "Positives directories  (add multiple shows / characters)",
            self.ml_train_positives,
        ))

        self.ml_train_sources = MultiPathList(accept_files=True, accept_dirs=True)
        vbox.addWidget(_field(
            "Source videos for negatives  (files and/or folders)",
            self.ml_train_sources,
        ))

        self.le_train_output = _make_line_edit(text="models/aesthetic.pt")
        btn_to = _browse_btn()
        btn_to.clicked.connect(self._browse_train_output)
        vbox.addWidget(_field("Save model to", self.le_train_output, btn_to))

        vbox.addWidget(_divider())

        # ── Parameters ────────────────────────────────────────────────────────
        vbox.addWidget(_section_header("Parameters"))

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 500)
        self.spin_epochs.setValue(20)
        vbox.addWidget(_spin_row("Epochs", self.spin_epochs))

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(2, 128)
        self.spin_batch.setValue(16)
        vbox.addWidget(_spin_row("Batch size", self.spin_batch))

        vbox.addStretch()

        # ── Action button ─────────────────────────────────────────────────────
        self.btn_train = QPushButton("Train Model")
        self.btn_train.setObjectName("primary")
        self.btn_train.setCursor(Qt.PointingHandCursor)
        self.btn_train.clicked.connect(self._start_training)
        vbox.addWidget(self.btn_train)

        return page

    def _build_bottom_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: #0a0a0a; border-top: 1px solid #151515;")

        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(20, 12, 20, 16)
        vbox.setSpacing(6)

        # Status / percentage row
        status_row = QWidget()
        h = QHBoxLayout(status_row)
        h.setContentsMargins(0, 0, 0, 0)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #555555; font-size: 12px;")
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h.addWidget(self.status_label)

        self.pct_label = QLabel("0%")
        self.pct_label.setStyleSheet("color: #444444; font-size: 11px;")
        self.pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(self.pct_label)

        vbox.addWidget(status_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        vbox.addWidget(self.progress_bar)

        # Log terminal
        self.log_view = LogView()
        self.log_view.setMinimumHeight(150)
        vbox.addWidget(self.log_view, stretch=1)

        return panel

    # ── Logging ───────────────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        handler = _ColorLogHandler(self.log_view)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    # ── Tab animation ─────────────────────────────────────────────────────────

    def _animate_tab_in(self, index: int) -> None:
        scroll = self.tabs.widget(index)
        if not scroll:
            return
        effect = QGraphicsOpacityEffect(scroll)
        scroll.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(180)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutQuad)
        anim.start(QAbstractAnimation.DeleteWhenStopped)
        # Keep reference alive until animation ends (Python GC)
        self._last_tab_anim = anim

    # ── Browse callbacks ──────────────────────────────────────────────────────

    def _browse_refs(self) -> None:
        ps, _ = QFileDialog.getOpenFileNames(
            self, "Select reference face images",
            "", "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*)"
        )
        if ps:
            self.le_refs.setText(";".join(ps))

    def _browse_library(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select clips library folder")
        if p:
            self.le_library.setText(p)

    def _browse_model(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select aesthetic model",
            "", "PyTorch model (*.pt *.pth);;All files (*)"
        )
        if p:
            self.le_model.setText(p)

    def _browse_output(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select output folder")
        if p:
            self.le_output.setText(p)

    def _browse_train_output(self) -> None:
        p, _ = QFileDialog.getSaveFileName(
            self, "Save model as",
            "models/aesthetic.pt", "PyTorch model (*.pt *.pth)"
        )
        if p:
            self.le_train_output.setText(p)

    # ── Action handlers ───────────────────────────────────────────────────────

    def _start_extraction(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "A job is already running.")
            return

        from autoclipper.constants import scan_for_videos, VIDEO_EXTENSIONS
        from autoclipper.pipeline import PipelineConfig

        if self.ml_input.is_empty():
            QMessageBox.warning(self, "No input", "Add at least one video file or folder.")
            return

        # Expand any folders to individual video files
        input_paths: list[Path] = []
        for p in self.ml_input.paths():
            if p.is_file():
                input_paths.append(p)
            else:
                input_paths.extend(scan_for_videos(p))

        if not input_paths:
            QMessageBox.warning(self, "No videos found",
                                "No supported video files were found in the selected paths.")
            return

        out_dir = Path(self.le_output.text().strip() or "output_clips")
        char = self.le_character.text().strip()

        base_cfg = PipelineConfig(
            input_video=input_paths[0],   # overridden per-file in worker
            character_name=char,
            output_dir=out_dir,
            min_motion_score=self.sld_motion.value(),
            min_aesthetic_score=self.sld_aesthetic.value(),
            scene_threshold=self.sld_scene.value(),
        )

        refs_text = self.le_refs.text().strip()
        if refs_text:
            base_cfg.reference_images = [Path(p) for p in refs_text.split(";") if p]

        lib = self.le_library.text().strip()
        if lib:
            base_cfg.clips_library_dir = Path(lib)

        model = self.le_model.text().strip()
        if model:
            base_cfg.model_path = Path(model)

        if char:
            base_cfg.face_db_path = out_dir / f"facedb_{char.replace(' ', '_')}.pkl"

        self._worker = ExtractionWorker(base_cfg, input_paths)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_extraction_done)
        self._worker.error.connect(self._on_error)
        self._set_busy(True)
        self._worker.start()

    def _start_training(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "A job is already running.")
            return

        positives_dirs = self.ml_train_positives.paths()
        source_paths = self.ml_train_sources.paths()
        output = self.le_train_output.text().strip() or "models/aesthetic.pt"

        if not positives_dirs or not source_paths:
            QMessageBox.warning(
                self, "Missing fields",
                "Add at least one positives folder and one source video or folder."
            )
            return

        self._worker = TrainingWorker(
            positives_dirs, source_paths, Path(output),
            self.spin_epochs.value(), self.spin_batch.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_training_done)
        self._worker.error.connect(self._on_error)
        self._set_busy(True)
        self._worker.start()

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_progress(self, message: str, pct: float) -> None:
        new_val = int(pct * 100)
        if self._prog_anim:
            self._prog_anim.stop()
            self._prog_anim.setStartValue(self.progress_bar.value())
            self._prog_anim.setEndValue(new_val)
            self._prog_anim.start()
        else:
            self.progress_bar.setValue(new_val)
        self.pct_label.setText(f"{new_val}%")
        self.status_label.setText(message)

    def _on_extraction_done(self, results: list) -> None:
        self._set_busy(False)
        n = len(results)
        out = self.le_output.text().strip()
        self._on_progress(f"Done. {n} clips → {out}", 1.0)
        QMessageBox.information(self, "Complete", f"Extracted {n} clips to:\n{out}")

    def _on_training_done(self, model_path: str) -> None:
        self._set_busy(False)
        self.le_model.setText(model_path)
        self._on_progress(f"Training complete. Saved → {model_path}", 1.0)
        QMessageBox.information(self, "Training Complete", f"Model saved to:\n{model_path}")

    def _on_error(self, msg: str) -> None:
        self._set_busy(False)
        self.status_label.setText(f"Error: {msg}")
        self.status_label.setStyleSheet("color: #ff4444; font-size: 12px;")
        QMessageBox.critical(self, "Error", msg)

    def _set_busy(self, busy: bool) -> None:
        self.btn_extract.setEnabled(not busy)
        self.btn_train.setEnabled(not busy)
        if not busy:
            self.status_label.setStyleSheet("color: #555555; font-size: 12px;")
        if busy:
            self.progress_bar.setValue(0)
            self.pct_label.setText("0%")
