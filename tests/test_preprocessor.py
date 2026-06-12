"""Tests for preprocessor module (no real ffmpeg required – mocked)."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoclipper.preprocessor import extract_clip, get_duration, preprocess


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


@patch("autoclipper.preprocessor.shutil.which", return_value=None)
def test_preprocess_raises_if_no_ffmpeg(mock_which, tmp_path):
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        preprocess(tmp_path / "in.mp4", tmp_path / "out.mp4")


@patch("subprocess.run")
def test_get_duration(mock_run):
    mock_run.return_value = MagicMock(stdout="123.456\n", returncode=0)
    dur = get_duration(Path("dummy.mp4"))
    assert abs(dur - 123.456) < 1e-6


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
