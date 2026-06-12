"""Tests for preprocessor module (no real ffmpeg required – mocked)."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoclipper.preprocessor import _ff, extract_clip, get_duration, preprocess


# ── _ff helper ────────────────────────────────────────────────────────────────

def test_ff_produces_file_uri(tmp_path):
    p = tmp_path / "normal.mp4"
    assert _ff(p).startswith("file:")
    assert "normal.mp4" in _ff(p)


def test_ff_special_chars_not_mangled(tmp_path):
    # brackets, parens, at-sign must survive verbatim so FFmpeg's file:
    # handler passes them straight to open()
    p = tmp_path / "Show [S01E01] (HD)@60fps.mp4"
    uri = _ff(p)
    assert uri.startswith("file:")
    assert "[S01E01]" in uri
    assert "(HD)" in uri
    assert "@60fps" in uri


# ── preprocess ────────────────────────────────────────────────────────────────

@patch("autoclipper.preprocessor.shutil.which", return_value="/usr/bin/ffmpeg")
@patch("autoclipper.preprocessor._run")
def test_preprocess_calls_ffmpeg(mock_run, mock_which, tmp_path):
    src = tmp_path / "input.mp4"
    src.touch()
    dst = tmp_path / "out.mp4"

    preprocess(src, dst)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "libx264" in cmd
    assert "-an" in cmd
    assert "24000/1001" in cmd


@patch("autoclipper.preprocessor.shutil.which", return_value="/usr/bin/ffmpeg")
@patch("autoclipper.preprocessor._run")
def test_preprocess_uses_file_uri(mock_run, mock_which, tmp_path):
    src = tmp_path / "Show [HD].mp4"
    src.touch()
    dst = tmp_path / "out.mp4"

    preprocess(src, dst)

    cmd = mock_run.call_args[0][0]
    # Both input and output must use file: URIs
    file_uris = [arg for arg in cmd if arg.startswith("file:")]
    assert len(file_uris) == 2, f"Expected 2 file: URIs in cmd, got: {file_uris}"
    assert any("Show [HD].mp4" in u for u in file_uris)


@patch("autoclipper.preprocessor.shutil.which", return_value=None)
def test_preprocess_raises_if_no_ffmpeg(mock_which, tmp_path):
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        preprocess(tmp_path / "in.mp4", tmp_path / "out.mp4")


# ── get_duration ──────────────────────────────────────────────────────────────

@patch("subprocess.run")
def test_get_duration(mock_run):
    mock_run.return_value = MagicMock(stdout="123.456\n", returncode=0)
    dur = get_duration(Path("dummy.mp4"))
    assert abs(dur - 123.456) < 1e-6


@patch("subprocess.run")
def test_get_duration_uses_file_uri(mock_run):
    mock_run.return_value = MagicMock(stdout="10.0\n", returncode=0)
    get_duration(Path("/movies/Show [S01].mp4"))
    args = mock_run.call_args[0][0]
    assert any(a.startswith("file:") and "[S01]" in a for a in args)


# ── extract_clip ──────────────────────────────────────────────────────────────

@patch("autoclipper.preprocessor.shutil.which", return_value="/usr/bin/ffmpeg")
@patch("autoclipper.preprocessor._run")
def test_extract_clip(mock_run, mock_which, tmp_path):
    src = tmp_path / "work.mp4"
    src.touch()
    dst = tmp_path / "clip.mp4"
    extract_clip(src, 5.0, 12.5, dst)

    cmd = mock_run.call_args[0][0]
    assert "-ss" in cmd
    assert "5.000000" in cmd
    assert "-t" in cmd
    assert "7.500000" in cmd


@patch("autoclipper.preprocessor.shutil.which", return_value="/usr/bin/ffmpeg")
@patch("autoclipper.preprocessor._run")
def test_extract_clip_uses_file_uri_for_special_chars(mock_run, mock_which, tmp_path):
    src = tmp_path / "Movie (2024) [4K].mkv"
    src.touch()
    dst = tmp_path / "out clip.mp4"

    extract_clip(src, 0.0, 5.0, dst)

    cmd = mock_run.call_args[0][0]
    file_uris = [arg for arg in cmd if arg.startswith("file:")]
    assert len(file_uris) == 2
    assert any("[4K]" in u for u in file_uris)
