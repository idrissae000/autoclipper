"""
Main extraction pipeline. Orchestrates:
  1. Preprocessing (FFmpeg)
  2. Scene / cut detection
  3. Motion scoring
  4. Character filtering (optional)
  5. Aesthetic scoring (optional, requires trained model)
  6. Clip export
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

ProgressCb = Callable[[str, float], None]   # (message, 0..1)


@dataclass
class PipelineConfig:
    # Input
    input_video: Path = Path()
    character_name: str = ""

    # Character database
    reference_images: list[Path] = field(default_factory=list)
    clips_library_dir: Path | None = None
    face_db_path: Path | None = None   # cached .pkl

    # Trained aesthetic model (optional)
    model_path: Path | None = None

    # Motion threshold to keep a scene
    min_motion_score: float = 1.2

    # Aesthetic model threshold (only applied when model_path set)
    min_aesthetic_score: float = 0.55

    # Output
    output_dir: Path = Path("output_clips")
    keep_preprocessed: bool = False

    # Scene detection
    scene_threshold: float = 27.0
    min_scene_len_frames: int = 15


@dataclass
class ExtractedClip:
    path: Path
    start_time: float
    end_time: float
    motion_score: float
    aesthetic_score: float = 0.0
    character_present: bool = True


def run(
    config: PipelineConfig,
    progress_cb: ProgressCb | None = None,
) -> list[ExtractedClip]:
    """
    Run the full extraction pipeline.

    Returns a list of extracted clips written to config.output_dir.
    """
    from autoclipper import preprocessor, scene_detector

    def cb(msg: str, pct: float):
        if progress_cb:
            progress_cb(msg, pct)
        log.info("[%.0f%%] %s", pct * 100, msg)

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    cb("Preprocessing video…", 0.0)
    with tempfile.TemporaryDirectory(prefix="autoclipper_") as tmpdir:
        tmp = Path(tmpdir)
        work_video = tmp / ("work_" + config.input_video.stem + ".mp4")

        preprocessor.preprocess(
            config.input_video,
            work_video,
            progress_cb=lambda p: cb("Preprocessing…", p * 0.20),
        )

        fps = 23.976

        # ── Step 2: Scene detection ───────────────────────────────────────────
        cb("Detecting scenes…", 0.20)
        scenes = scene_detector.detect_scenes(
            work_video,
            fps=fps,
            threshold=config.scene_threshold,
            min_scene_len_frames=config.min_scene_len_frames,
            progress_cb=lambda p: cb("Detecting scenes…", 0.20 + p * 0.15),
        )
        cb(f"Found {len(scenes)} candidate scenes", 0.35)

        # ── Step 3: Motion scoring ────────────────────────────────────────────
        cb("Scoring motion…", 0.35)
        scenes = scene_detector.score_motion(
            work_video,
            scenes,
            progress_cb=lambda p: cb("Scoring motion…", 0.35 + p * 0.15),
        )
        before = len(scenes)
        scenes = [s for s in scenes if s.motion_score >= config.min_motion_score]
        cb(f"After motion filter: {len(scenes)}/{before} scenes kept", 0.50)

        # ── Step 4: Character filter (optional) ───────────────────────────────
        if config.character_name:
            scenes = _filter_by_character(scenes, work_video, config, cb)

        # ── Step 5: Aesthetic scoring (optional) ──────────────────────────────
        if config.model_path and config.model_path.exists():
            scenes = _filter_by_aesthetic(scenes, work_video, config, cb)

        # ── Step 6: Export clips ──────────────────────────────────────────────
        results = _export_clips(scenes, work_video, config, cb)

        # Optionally keep the preprocessed file
        if config.keep_preprocessed:
            dest = config.output_dir / ("preprocessed_" + config.input_video.stem + ".mp4")
            shutil.copy2(work_video, dest)

    cb(f"Done. Extracted {len(results)} clips → {config.output_dir}", 1.0)
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _filter_by_character(scenes, work_video, config: PipelineConfig, cb) -> list:
    from autoclipper.face_recognition import FaceDatabase, CharacterDetector

    cb("Loading character database…", 0.50)
    db = FaceDatabase()

    if config.face_db_path and config.face_db_path.exists():
        db.load(config.face_db_path)
    elif config.reference_images:
        detector_tmp = CharacterDetector(db, config.character_name)
        detector_tmp.build_db_from_images(config.reference_images)
    elif config.clips_library_dir and config.clips_library_dir.exists():
        detector_tmp = CharacterDetector(db, config.character_name)
        detector_tmp.build_db_from_clips(config.clips_library_dir)
    else:
        log.warning("No face database source – skipping character filter")
        return scenes

    if config.face_db_path:
        db.save(config.face_db_path)

    if not db.has(config.character_name):
        log.warning("Character '%s' not in database – skipping filter", config.character_name)
        return scenes

    detector = CharacterDetector(db, config.character_name)
    filtered = []
    total = len(scenes)
    for i, scene in enumerate(scenes):
        cb(f"Character detection {i+1}/{total}…", 0.50 + (i / total) * 0.20)
        if detector.scene_has_character(work_video, scene.start_frame, scene.end_frame):
            filtered.append(scene)

    cb(f"Character filter: {len(filtered)}/{total} scenes kept", 0.70)
    return filtered


def _filter_by_aesthetic(scenes, work_video, config: PipelineConfig, cb) -> list:
    from autoclipper.clip_scorer import ClipScorerInference, sample_frames

    cb("Loading aesthetic model…", 0.70)
    scorer = ClipScorerInference(config.model_path)  # type: ignore[arg-type]
    filtered = []
    total = len(scenes)
    cap = __import__("cv2").VideoCapture(str(work_video))
    fps = cap.get(__import__("cv2").CAP_PROP_FPS) or 23.976

    for i, scene in enumerate(scenes):
        cb(f"Aesthetic scoring {i+1}/{total}…", 0.70 + (i / total) * 0.10)
        frames = _grab_frames(cap, scene.start_frame, scene.end_frame, n=8)
        score = scorer.score_frames(frames)
        scene.aesthetic_score = score
        if score >= config.min_aesthetic_score:
            filtered.append(scene)

    cap.release()
    cb(f"Aesthetic filter: {len(filtered)}/{total} scenes kept", 0.80)
    return filtered


def _grab_frames(cap, start_frame: int, end_frame: int, n: int = 8):
    import numpy as np
    import cv2

    total = end_frame - start_frame
    indices = [start_frame + int(i * total / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    return frames


def _export_clips(scenes, work_video, config: PipelineConfig, cb) -> list[ExtractedClip]:
    from autoclipper.preprocessor import extract_clip

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(scenes)
    stem = config.input_video.stem
    char = config.character_name.replace(" ", "_") if config.character_name else "clip"

    for i, scene in enumerate(scenes):
        cb(f"Exporting clip {i+1}/{total}…", 0.80 + (i / max(total, 1)) * 0.20)
        fname = f"{char}__{stem}__{i+1:04d}__{scene.start_time:.1f}-{scene.end_time:.1f}s.mp4"
        out_path = config.output_dir / fname
        try:
            extract_clip(work_video, scene.start_time, scene.end_time, out_path)
            results.append(ExtractedClip(
                path=out_path,
                start_time=scene.start_time,
                end_time=scene.end_time,
                motion_score=scene.motion_score,
                aesthetic_score=getattr(scene, "aesthetic_score", 0.0),
            ))
        except Exception as e:
            log.warning("Failed to export clip %d: %s", i + 1, e)

    return results
