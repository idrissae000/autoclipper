"""Shared constants for video file handling."""

from __future__ import annotations

from pathlib import Path

# All video formats accepted throughout the app
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".avi"})

# Qt file dialog filter string
VIDEO_FILTER = "Video files (*.mp4 *.mov *.m4v *.mkv *.avi);;All files (*)"


def scan_for_videos(root: Path) -> list[Path]:
    """Recursively collect every video file under *root*."""
    found: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        found.extend(root.rglob(f"*{ext}"))
    return sorted(set(found))
