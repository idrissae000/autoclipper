"""Tests for trainer.collect_positives and trainer.resolve_source_videos."""

from pathlib import Path
import pytest

from autoclipper.trainer import collect_positives, resolve_source_videos


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


# ── resolve_source_videos ─────────────────────────────────────────────────────

def test_single_file_accepted(tmp_path):
    f = _touch(tmp_path / "movie.mp4")
    result = resolve_source_videos([f])
    assert result == [f]


def test_m4v_accepted(tmp_path):
    f = _touch(tmp_path / "clip.m4v")
    result = resolve_source_videos([f])
    assert result == [f]


def test_single_file_wrong_ext_raises(tmp_path):
    f = _touch(tmp_path / "document.pdf")
    with pytest.raises(ValueError, match="Not a recognised video"):
        resolve_source_videos([f])


def test_folder_scans_recursively(tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "sub" / "b.mkv")
    _touch(tmp_path / "sub" / "ignore.txt")
    result = resolve_source_videos([tmp_path])
    names = {p.name for p in result}
    assert names == {"a.mp4", "b.mkv"}


def test_empty_folder_returns_empty(tmp_path):
    result = resolve_source_videos([tmp_path])
    assert result == []


def test_all_supported_exts(tmp_path):
    for ext in (".mp4", ".mov", ".m4v", ".mkv", ".avi"):
        _touch(tmp_path / f"clip{ext}")
    result = resolve_source_videos([tmp_path])
    assert len(result) == 5


def test_mix_of_files_and_folders(tmp_path):
    f1 = _touch(tmp_path / "standalone.mp4")
    sub = tmp_path / "season1"
    f2 = _touch(sub / "ep01.mkv")
    f3 = _touch(sub / "ep02.mov")

    result = resolve_source_videos([f1, sub])
    assert set(result) == {f1, f2, f3}


def test_deduplicates_across_sources(tmp_path):
    f = _touch(tmp_path / "movie.mp4")
    # same file added twice (once as file, once via parent folder)
    result = resolve_source_videos([f, tmp_path])
    assert result.count(f) == 1


# ── collect_positives ─────────────────────────────────────────────────────────

def test_collect_positives_single_dir(tmp_path):
    _touch(tmp_path / "clip1.mp4")
    _touch(tmp_path / "sub" / "clip2.mkv")
    result = collect_positives([tmp_path])
    assert len(result) == 2


def test_collect_positives_multiple_dirs(tmp_path):
    dir_a = tmp_path / "show_a"
    dir_b = tmp_path / "show_b"
    _touch(dir_a / "c1.mp4")
    _touch(dir_b / "c2.mov")
    _touch(dir_b / "c3.m4v")
    result = collect_positives([dir_a, dir_b])
    assert len(result) == 3


def test_collect_positives_deduplicates(tmp_path):
    _touch(tmp_path / "clip.mp4")
    # same dir listed twice
    result = collect_positives([tmp_path, tmp_path])
    assert len(result) == 1
