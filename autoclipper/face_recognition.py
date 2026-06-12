"""
Character face recognition using InsightFace (ArcFace backbone).

Workflow
--------
1. Build a face database from reference images or existing clip folders.
2. For a given video frame, detect all faces, embed them, find nearest match.
3. Return whether the target character appears in a frame / scene.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.45   # cosine distance – lower = more strict


class FaceDatabase:
    """Stores mean ArcFace embeddings for one or more characters."""

    def __init__(self):
        self._entries: dict[str, np.ndarray] = {}   # name → mean embedding

    # ── building ─────────────────────────────────────────────────────────────

    def add_from_images(self, name: str, image_paths: list[Path], analyzer) -> int:
        """Embed *image_paths* (reference photos) and store mean for *name*."""
        embeddings = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                log.warning("Could not read %s", p)
                continue
            faces = analyzer.get(img)
            if not faces:
                log.warning("No face detected in %s", p)
                continue
            # pick the largest face if multiple
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embeddings.append(face.normed_embedding)

        if not embeddings:
            log.error("No usable embeddings from images for '%s'", name)
            return 0

        self._entries[name] = np.mean(embeddings, axis=0)
        log.info("Added '%s': %d images → 1 mean embedding", name, len(embeddings))
        return len(embeddings)

    def add_from_clips(self, name: str, clip_dir: Path, analyzer, *, max_frames: int = 200) -> int:
        """
        Sample frames from mp4 files in *clip_dir* to build embeddings for *name*.

        Samples up to *max_frames* frames total across all clips.
        """
        clips = sorted(clip_dir.rglob("*.mp4"))
        if not clips:
            log.warning("No mp4 files found in %s", clip_dir)
            return 0

        embeddings = []
        frames_per_clip = max(1, max_frames // len(clips))

        for clip_path in clips:
            cap = cv2.VideoCapture(str(clip_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total // frames_per_clip)
            fno = 0
            sampled = 0

            while sampled < frames_per_clip:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                ret, frame = cap.read()
                if not ret:
                    break
                faces = analyzer.get(frame)
                for face in faces:
                    embeddings.append(face.normed_embedding)
                fno += step
                sampled += 1

            cap.release()

        if not embeddings:
            log.error("No faces found in clips for '%s'", name)
            return 0

        self._entries[name] = np.mean(embeddings, axis=0)
        log.info("Added '%s' from %d clips → mean of %d embeddings", name, len(clips), len(embeddings))
        return len(embeddings)

    # ── lookup ────────────────────────────────────────────────────────────────

    def find(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return (name, distance) of closest match, or (None, 1.0) if DB empty."""
        if not self._entries:
            return None, 1.0
        best_name = None
        best_dist = float("inf")
        for name, ref in self._entries.items():
            dist = float(1.0 - float(np.dot(embedding, ref)))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name, best_dist

    def has(self, name: str) -> bool:
        return name in self._entries

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._entries, f)
        log.info("Saved face database to %s", path)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self._entries = pickle.load(f)
        log.info("Loaded face database from %s (%d entries)", path, len(self._entries))


class CharacterDetector:
    """
    Detects whether a target character appears in video frames.

    Uses InsightFace for face detection + ArcFace embedding.
    """

    def __init__(self, db: FaceDatabase, target_name: str):
        self.db = db
        self.target = target_name
        self._analyzer = None

    def _get_analyzer(self):
        if self._analyzer is None:
            try:
                import insightface
                from insightface.app import FaceAnalysis
                self._analyzer = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._analyzer.prepare(ctx_id=0, det_size=(640, 640))
            except ImportError:
                raise RuntimeError(
                    "insightface not installed. Run: pip install insightface onnxruntime-gpu"
                )
        return self._analyzer

    def character_present(self, frame: np.ndarray) -> bool:
        """Return True if the target character is detected in *frame*."""
        analyzer = self._get_analyzer()
        faces = analyzer.get(frame)
        if not faces:
            return False
        for face in faces:
            name, dist = self.db.find(face.normed_embedding)
            if name == self.target and dist < SIMILARITY_THRESHOLD:
                return True
        return False

    def scene_has_character(
        self,
        video_path: Path,
        start_frame: int,
        end_frame: int,
        *,
        min_hit_fraction: float = 0.3,
        sample_fps: float = 2.0,
    ) -> bool:
        """
        Return True if the character appears in >= *min_hit_fraction* of sampled frames.
        """
        cap = cv2.VideoCapture(str(video_path))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 23.976
        step = max(1, int(actual_fps / sample_fps))

        total_sampled = 0
        hits = 0
        fno = start_frame

        while fno <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = cap.read()
            if not ret:
                break
            if self.character_present(frame):
                hits += 1
            total_sampled += 1
            fno += step

        cap.release()

        if total_sampled == 0:
            return False
        fraction = hits / total_sampled
        log.debug("Scene [%d-%d]: character hit %.1f%%", start_frame, end_frame, fraction * 100)
        return fraction >= min_hit_fraction

    def build_db_from_images(self, image_paths: list[Path]) -> None:
        analyzer = self._get_analyzer()
        self.db.add_from_images(self.target, image_paths, analyzer)

    def build_db_from_clips(self, clip_dir: Path) -> None:
        analyzer = self._get_analyzer()
        self.db.add_from_clips(self.target, clip_dir, analyzer)
