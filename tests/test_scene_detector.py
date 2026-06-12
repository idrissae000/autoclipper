"""Tests for scene detector – uses a synthetic in-memory video."""

import numpy as np
import pytest

from autoclipper.scene_detector import SceneBoundary, score_motion


def _make_fake_video(tmp_path, n_frames=60, width=320, height=180, fps=24.0):
    """Write a minimal synthetic .mp4 using OpenCV VideoWriter."""
    import cv2

    path = tmp_path / "fake.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    rng = np.random.default_rng(0)
    for i in range(n_frames):
        # Hard cut at frame 30: switch from dark to bright
        if i < 30:
            frame = np.full((height, width, 3), 20, dtype=np.uint8)
        else:
            frame = np.full((height, width, 3), 220, dtype=np.uint8)
        # Add random noise
        noise = rng.integers(0, 30, (height, width, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return path


def test_score_motion_returns_non_negative(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = _make_fake_video(tmp_path, n_frames=48)

    boundaries = [
        SceneBoundary(start_frame=0, end_frame=23, start_time=0.0, end_time=1.0),
        SceneBoundary(start_frame=24, end_frame=47, start_time=1.0, end_time=2.0),
    ]

    result = score_motion(path, boundaries, sample_fps=4.0)
    for b in result:
        assert b.motion_score >= 0.0


def test_scene_boundary_duration():
    b = SceneBoundary(start_frame=0, end_frame=48, start_time=1.5, end_time=4.0)
    assert abs(b.duration - 2.5) < 1e-9
