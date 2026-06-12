"""Scene / cut boundary detection using PySceneDetect + motion analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Minimum / maximum clip length in seconds
MIN_CLIP_SECS = 1.5
MAX_CLIP_SECS = 30.0


@dataclass
class SceneBoundary:
    start_frame: int
    end_frame: int
    start_time: float   # seconds
    end_time: float     # seconds
    motion_score: float = 0.0
    scene_score: float = 0.0  # cut sharpness

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def detect_scenes(
    video_path: Path,
    fps: float = 23.976,
    *,
    threshold: float = 27.0,
    min_scene_len_frames: int = 15,
    progress_cb=None,
) -> list[SceneBoundary]:
    """
    Detect scene boundaries in *video_path*.

    Uses PySceneDetect's ContentDetector (HSV histogram delta) and falls back
    to a pure-OpenCV frame-difference detector if scenedetect is unavailable.

    Returns list of SceneBoundary sorted by start time.
    """
    try:
        return _detect_scenedetect(video_path, fps, threshold, min_scene_len_frames, progress_cb)
    except ImportError:
        log.warning("PySceneDetect not installed – using fallback detector")
        return _detect_opencv(video_path, fps, threshold, min_scene_len_frames, progress_cb)


# ── PySceneDetect backend ────────────────────────────────────────────────────

def _detect_scenedetect(
    video_path: Path,
    fps: float,
    threshold: float,
    min_scene_len_frames: int,
    progress_cb,
) -> list[SceneBoundary]:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames))

    manager.detect_scenes(video, show_progress=False, callback=_make_sd_callback(progress_cb))

    raw = manager.get_scene_list()
    boundaries = []
    for start_tc, end_tc in raw:
        sf = start_tc.get_frames()
        ef = end_tc.get_frames()
        st = start_tc.get_seconds()
        et = end_tc.get_seconds()
        dur = et - st
        if dur < MIN_CLIP_SECS or dur > MAX_CLIP_SECS:
            continue
        boundaries.append(SceneBoundary(sf, ef, st, et))

    log.info("SceneDetect found %d candidate scenes in %s", len(boundaries), video_path.name)
    return boundaries


def _make_sd_callback(progress_cb):
    if progress_cb is None:
        return None
    def cb(frame_num, frame_rate, total_frames):
        if total_frames:
            progress_cb(frame_num / total_frames)
    return cb


# ── OpenCV fallback backend ──────────────────────────────────────────────────

def _detect_opencv(
    video_path: Path,
    fps: float,
    threshold: float,
    min_scene_len_frames: int,
    progress_cb,
) -> list[SceneBoundary]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    cuts: list[int] = [0]
    prev_hsv = None
    fno = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if prev_hsv is not None:
            diff = cv2.compareHist(prev_hsv, hist, cv2.HISTCMP_CHISQR)
            if diff > threshold and (fno - cuts[-1]) >= min_scene_len_frames:
                cuts.append(fno)

        prev_hsv = hist
        fno += 1

        if progress_cb and total > 0 and fno % 100 == 0:
            progress_cb(fno / total)

    cuts.append(fno)
    cap.release()

    boundaries = []
    for i in range(len(cuts) - 1):
        sf = cuts[i]
        ef = cuts[i + 1]
        st = sf / actual_fps
        et = ef / actual_fps
        if (et - st) < MIN_CLIP_SECS or (et - st) > MAX_CLIP_SECS:
            continue
        boundaries.append(SceneBoundary(sf, ef, st, et))

    log.info("OpenCV detector found %d candidate scenes", len(boundaries))
    return boundaries


# ── Motion scoring ────────────────────────────────────────────────────────────

def score_motion(
    video_path: Path,
    boundaries: list[SceneBoundary],
    *,
    sample_fps: float = 4.0,
    progress_cb=None,
) -> list[SceneBoundary]:
    """
    Compute dense optical-flow motion score for each boundary.

    Samples the clip at *sample_fps* and accumulates mean flow magnitude.
    Modifies boundaries in-place (motion_score field) and returns them.
    """
    cap = cv2.VideoCapture(str(video_path))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 23.976
    step = max(1, int(actual_fps / sample_fps))
    total_boundaries = len(boundaries)

    for idx, b in enumerate(boundaries):
        cap.set(cv2.CAP_PROP_POS_FRAMES, b.start_frame)
        scores = []
        prev_gray = None
        fno = b.start_frame

        while fno <= b.end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (fno - b.start_frame) % step == 0:
                small = cv2.resize(frame, (320, 180))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray,
                        None, 0.5, 3, 15, 3, 5, 1.2, 0,
                    )
                    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    scores.append(float(mag.mean()))
                prev_gray = gray
            fno += 1

        b.motion_score = float(np.mean(scores)) if scores else 0.0

        if progress_cb:
            progress_cb((idx + 1) / total_boundaries)

    cap.release()
    return boundaries
