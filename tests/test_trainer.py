"""Tests for trainer.resolve_source_videos."""

from pathlib import Path
import pytest

from autoclipper.trainer import resolve_source_videos


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_single_file_accepted(tmp_path):
    f = _touch(tmp_path / "movie.mp4")
    result = resolve_source_videos(f)
    assert result == [f]


def test_single_file_wrong_ext_raises(tmp_path):
    f = _touch(tmp_path / "document.pdf")
    with pytest.raises(ValueError, match="Not a recognised video"):
        resolve_source_videos(f)


def test_folder_scans_recursively(tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "sub" / "b.mkv")
    _touch(tmp_path / "sub" / "ignore.txt")
    result = resolve_source_videos(tmp_path)
    names = {p.name for p in result}
    assert names == {"a.mp4", "b.mkv"}


def test_empty_folder_returns_empty(tmp_path):
    result = resolve_source_videos(tmp_path)
    assert result == []


def test_folder_with_all_supported_exts(tmp_path):
    exts = [".mp4", ".mkv", ".mov", ".avi", ".ts"]
    for ext in exts:
        _touch(tmp_path / f"clip{ext}")
    result = resolve_source_videos(tmp_path)
    assert len(result) == len(exts)
