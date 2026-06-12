"""
AutoClipper desktop UI built with PyQt5.

Layout:
  ┌─────────────────────────────────────────────┐
  │  [Input Video]  [Browse]                    │
  │  [Character name]                           │
  │  [Ref images] [Browse]  ─ OR ─              │
  │  [Clips library] [Browse]                   │
  │  [Model (optional)] [Browse]                │
  │  [Output folder] [Browse]                   │
  │  ─────────── Advanced ───────────           │
  │  Motion threshold  [1.2]                    │
  │  Aesthetic thresh  [0.55]                   │
  │                                             │
  │  [  Train Model  ]   [  Extract Clips  ]    │
  │                                             │
  │  ████████████████░░░░  42%                  │
  │  Log output …                               │
  └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QSpinBox,
)

log = logging.getLogger(__name__)


# ── Worker threads ────────────────────────────────────────────────────────────

class ExtractionWorker(QThread):
    progress = pyqtSignal(str, float)    # message, 0..1
    finished = pyqtSignal(list)          # list of ExtractedClip
    error = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            from autoclipper.pipeline import run
            results = run(self.config, progress_cb=lambda m, p: self.progress.emit(m, p))
            self.finished.emit(results)
        except Exception as e:
            log.exception("Extraction failed")
            self.error.emit(str(e))


class TrainingWorker(QThread):
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, positives_dir, sources_dir, output_model, epochs, batch_size):
        super().__init__()
        self.positives_dir = positives_dir
        self.sources_dir = sources_dir
        self.output_model = output_model
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        try:
            from autoclipper.trainer import build_and_train
            build_and_train(
                self.positives_dir,
                self.sources_dir,
                self.output_model,
                epochs=self.epochs,
                batch_size=self.batch_size,
                progress_cb=lambda m, p: self.progress.emit(m, p),
            )
            self.finished.emit(str(self.output_model))
        except Exception as e:
            log.exception("Training failed")
            self.error.emit(str(e))


# ── Qt log handler ────────────────────────────────────────────────────────────

class _QtLogHandler(logging.Handler):
    def __init__(self, widget: QPlainTextEdit):
        super().__init__()
        self._widget = widget

    def emit(self, record):
        msg = self.format(record)
        # Must dispatch to main thread
        from PyQt5.QtCore import QMetaObject, Q_ARG
        QMetaObject.invokeMethod(
            self._widget, "appendPlainText",
            Qt.QueuedConnection,
            Q_ARG(str, msg),
        )


# ── Widgets ───────────────────────────────────────────────────────────────────

def _browse_file(parent, caption, filter=""):
    path, _ = QFileDialog.getOpenFileName(parent, caption, "", filter)
    return path


def _browse_files(parent, caption, filter=""):
    paths, _ = QFileDialog.getOpenFileNames(parent, caption, "", filter)
    return paths


def _browse_dir(parent, caption):
    return QFileDialog.getExistingDirectory(parent, caption)


def _path_row(parent, label: str, line_edit: QLineEdit, btn_label="Browse…") -> QWidget:
    """Return a horizontal widget: label + line_edit + button."""
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(line_edit)
    btn = QPushButton(btn_label)
    btn.setFixedWidth(90)
    h.addWidget(btn)
    return row, btn


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoClipper")
        self.setMinimumSize(780, 700)
        self._worker: QThread | None = None
        self._setup_ui()
        self._setup_logging()

    # ── UI setup ──────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(10)
        vbox.setContentsMargins(14, 14, 14, 14)

        # Title
        title = QLabel("AutoClipper")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        vbox.addWidget(title)

        # ── Extraction form ───────────────────────────────────────────────────
        extract_group = QGroupBox("Clip Extraction")
        form = QFormLayout(extract_group)
        form.setSpacing(8)

        self.le_input = QLineEdit()
        self.le_input.setPlaceholderText("Select a video file…")
        row_input, btn_input = _path_row(self, "Input video", self.le_input)
        btn_input.clicked.connect(self._browse_input)
        form.addRow("Input video:", row_input)

        self.le_character = QLineEdit()
        self.le_character.setPlaceholderText("e.g. Walter White  (leave blank to skip)")
        form.addRow("Character name:", self.le_character)

        self.le_refs = QLineEdit()
        self.le_refs.setPlaceholderText("Reference images (5–10 face photos)")
        row_refs, btn_refs = _path_row(self, "Ref images", self.le_refs)
        btn_refs.clicked.connect(self._browse_refs)
        form.addRow("Ref images:", row_refs)

        self.le_library = QLineEdit()
        self.le_library.setPlaceholderText("Or point to existing clip folder for this character")
        row_lib, btn_lib = _path_row(self, "Clips library", self.le_library)
        btn_lib.clicked.connect(self._browse_library)
        form.addRow("Clips library:", row_lib)

        self.le_model = QLineEdit()
        self.le_model.setPlaceholderText("Optional: trained aesthetic model (.pt)")
        row_model, btn_model = _path_row(self, "Model", self.le_model)
        btn_model.clicked.connect(self._browse_model)
        form.addRow("Aesthetic model:", row_model)

        self.le_output = QLineEdit()
        self.le_output.setText("output_clips")
        row_out, btn_out = _path_row(self, "Output folder", self.le_output)
        btn_out.clicked.connect(self._browse_output)
        form.addRow("Output folder:", row_out)

        vbox.addWidget(extract_group)

        # ── Advanced settings ─────────────────────────────────────────────────
        adv_group = QGroupBox("Advanced")
        adv_form = QFormLayout(adv_group)
        adv_group.setCheckable(True)
        adv_group.setChecked(False)

        self.spin_motion = QDoubleSpinBox()
        self.spin_motion.setRange(0.1, 20.0)
        self.spin_motion.setSingleStep(0.1)
        self.spin_motion.setValue(1.2)
        adv_form.addRow("Min motion score:", self.spin_motion)

        self.spin_aesthetic = QDoubleSpinBox()
        self.spin_aesthetic.setRange(0.1, 1.0)
        self.spin_aesthetic.setSingleStep(0.05)
        self.spin_aesthetic.setValue(0.55)
        adv_form.addRow("Min aesthetic score:", self.spin_aesthetic)

        self.spin_scene_thresh = QDoubleSpinBox()
        self.spin_scene_thresh.setRange(5.0, 100.0)
        self.spin_scene_thresh.setSingleStep(1.0)
        self.spin_scene_thresh.setValue(27.0)
        adv_form.addRow("Scene cut threshold:", self.spin_scene_thresh)

        vbox.addWidget(adv_group)

        # ── Training form ─────────────────────────────────────────────────────
        train_group = QGroupBox("Train Aesthetic Model")
        train_form = QFormLayout(train_group)

        self.le_train_positives = QLineEdit()
        self.le_train_positives.setPlaceholderText("Folder containing your 4000+ positive clips")
        row_tp, btn_tp = _path_row(self, "Positives", self.le_train_positives)
        btn_tp.clicked.connect(self._browse_train_positives)
        train_form.addRow("Positives dir:", row_tp)

        self.le_train_sources = QLineEdit()
        self.le_train_sources.setPlaceholderText("Raw source videos folder (for negative generation)")
        row_ts, btn_ts = _path_row(self, "Source videos", self.le_train_sources)
        btn_ts.clicked.connect(self._browse_train_sources)
        train_form.addRow("Source videos:", row_ts)

        self.le_train_output = QLineEdit()
        self.le_train_output.setText("models/aesthetic.pt")
        row_to, btn_to = _path_row(self, "Model output", self.le_train_output)
        btn_to.clicked.connect(self._browse_train_output)
        train_form.addRow("Save model to:", row_to)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 200)
        self.spin_epochs.setValue(20)
        train_form.addRow("Epochs:", self.spin_epochs)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(2, 128)
        self.spin_batch.setValue(16)
        train_form.addRow("Batch size:", self.spin_batch)

        vbox.addWidget(train_group)

        # ── Action buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_train = QPushButton("Train Model")
        self.btn_train.setFixedHeight(38)
        self.btn_train.clicked.connect(self._start_training)
        btn_row.addWidget(self.btn_train)

        self.btn_extract = QPushButton("Extract Clips")
        self.btn_extract.setFixedHeight(38)
        self.btn_extract.setDefault(True)
        self.btn_extract.clicked.connect(self._start_extraction)
        btn_row.addWidget(self.btn_extract)
        vbox.addLayout(btn_row)

        # ── Progress ──────────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        vbox.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        vbox.addWidget(self.status_label)

        # ── Log output ────────────────────────────────────────────────────────
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFont(QFont("Monospace", 8))
        self.log_view.setMinimumHeight(140)
        vbox.addWidget(self.log_view)

    def _setup_logging(self):
        handler = _QtLogHandler(self.log_view)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    # ── Browse callbacks ──────────────────────────────────────────────────────

    def _browse_input(self):
        p = _browse_file(self, "Select input video", "Video files (*.mp4 *.mkv *.mov *.avi *.ts);;All files (*)")
        if p:
            self.le_input.setText(p)

    def _browse_refs(self):
        ps = _browse_files(self, "Select reference face images", "Images (*.jpg *.jpeg *.png *.webp);;All files (*)")
        if ps:
            self.le_refs.setText(";".join(ps))

    def _browse_library(self):
        p = _browse_dir(self, "Select clips library folder")
        if p:
            self.le_library.setText(p)

    def _browse_model(self):
        p = _browse_file(self, "Select trained model", "PyTorch model (*.pt *.pth);;All files (*)")
        if p:
            self.le_model.setText(p)

    def _browse_output(self):
        p = _browse_dir(self, "Select output folder")
        if p:
            self.le_output.setText(p)

    def _browse_train_positives(self):
        p = _browse_dir(self, "Select positives clips folder")
        if p:
            self.le_train_positives.setText(p)

    def _browse_train_sources(self):
        p = _browse_dir(self, "Select source videos folder")
        if p:
            self.le_train_sources.setText(p)

    def _browse_train_output(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save model as", "models/aesthetic.pt", "PyTorch model (*.pt *.pth)")
        if p:
            self.le_train_output.setText(p)

    # ── Action handlers ───────────────────────────────────────────────────────

    def _start_extraction(self):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "A job is already running.")
            return

        from autoclipper.pipeline import PipelineConfig

        input_video = self.le_input.text().strip()
        if not input_video:
            QMessageBox.warning(self, "No input", "Please select an input video file.")
            return

        cfg = PipelineConfig(
            input_video=Path(input_video),
            character_name=self.le_character.text().strip(),
            output_dir=Path(self.le_output.text().strip() or "output_clips"),
            min_motion_score=self.spin_motion.value(),
            min_aesthetic_score=self.spin_aesthetic.value(),
            scene_threshold=self.spin_scene_thresh.value(),
        )

        refs_text = self.le_refs.text().strip()
        if refs_text:
            cfg.reference_images = [Path(p) for p in refs_text.split(";") if p]

        lib_text = self.le_library.text().strip()
        if lib_text:
            cfg.clips_library_dir = Path(lib_text)

        model_text = self.le_model.text().strip()
        if model_text:
            cfg.model_path = Path(model_text)

        # Cache face db next to output
        if cfg.character_name:
            cfg.face_db_path = cfg.output_dir / f"facedb_{cfg.character_name.replace(' ', '_')}.pkl"

        self._worker = ExtractionWorker(cfg)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_extraction_done)
        self._worker.error.connect(self._on_error)
        self._set_buttons_enabled(False)
        self.progress_bar.setValue(0)
        self._worker.start()

    def _start_training(self):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "A job is already running.")
            return

        positives = self.le_train_positives.text().strip()
        sources = self.le_train_sources.text().strip()
        output = self.le_train_output.text().strip() or "models/aesthetic.pt"

        if not positives or not sources:
            QMessageBox.warning(self, "Missing fields", "Positives folder and source videos folder are required.")
            return

        self._worker = TrainingWorker(
            Path(positives),
            Path(sources),
            Path(output),
            self.spin_epochs.value(),
            self.spin_batch.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_training_done)
        self._worker.error.connect(self._on_error)
        self._set_buttons_enabled(False)
        self.progress_bar.setValue(0)
        self._worker.start()

    # ── Worker signal handlers ────────────────────────────────────────────────

    def _on_progress(self, message: str, pct: float):
        self.progress_bar.setValue(int(pct * 100))
        self.status_label.setText(message)

    def _on_extraction_done(self, results):
        self._set_buttons_enabled(True)
        self.progress_bar.setValue(100)
        n = len(results)
        out = self.le_output.text().strip()
        self.status_label.setText(f"Done. {n} clips saved to {out}")
        QMessageBox.information(self, "Complete", f"Extracted {n} clips to:\n{out}")

    def _on_training_done(self, model_path: str):
        self._set_buttons_enabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Training complete. Model saved to {model_path}")
        self.le_model.setText(model_path)
        QMessageBox.information(self, "Training Complete", f"Model saved to:\n{model_path}")

    def _on_error(self, msg: str):
        self._set_buttons_enabled(True)
        self.status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def _set_buttons_enabled(self, enabled: bool):
        self.btn_extract.setEnabled(enabled)
        self.btn_train.setEnabled(enabled)
