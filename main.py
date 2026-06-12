#!/usr/bin/env python3
"""AutoClipper – launch the desktop app."""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    from PyQt5.QtWidgets import QApplication
    from autoclipper.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("AutoClipper")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
