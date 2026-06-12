"""
Training entry point: scan positive clip library, generate negatives, train model.

Usage (CLI):
  python -m autoclipper.trainer \
      --positives /path/to/clips \
      --sources   /path/to/raw_footage \
      --output    models/aesthetic.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def collect_positives(clips_root: Path) -> list[Path]:
    """Recursively collect all mp4 files under *clips_root* as positive examples."""
    clips = sorted(clips_root.rglob("*.mp4"))
    log.info("Found %d positive clips in %s", len(clips), clips_root)
    return clips


_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".ts"}


def resolve_source_videos(source_path: Path) -> list[Path]:
    """Accept a single video file or a directory; return a list of video paths."""
    if source_path.is_file():
        if source_path.suffix.lower() not in _VIDEO_EXTS:
            raise ValueError(f"Not a recognised video file: {source_path}")
        return [source_path]
    videos = []
    for ext in _VIDEO_EXTS:
        videos.extend(source_path.rglob(f"*{ext}"))
    return sorted(videos)


def build_and_train(
    positives_dir: Path,
    source_videos_path: Path,
    output_model: Path,
    *,
    negatives_dir: Path | None = None,
    epochs: int = 20,
    batch_size: int = 16,
    progress_cb=None,
):
    from autoclipper.clip_scorer import generate_negatives, train

    positives = collect_positives(positives_dir)
    if not positives:
        raise ValueError(f"No positive clips found in {positives_dir}")

    neg_dir = negatives_dir or output_model.parent / "negatives"
    source_videos = resolve_source_videos(source_videos_path)

    if not source_videos:
        raise ValueError(f"No source videos found in {source_videos_path}")

    log.info("Generating negative examples from %d source videos…", len(source_videos))
    negatives = generate_negatives(
        source_videos,
        positives,
        neg_dir,
        target_count=min(len(positives), 3000),
    )

    log.info("Training on %d positives + %d negatives", len(positives), len(negatives))
    train(
        positives,
        negatives,
        output_model,
        epochs=epochs,
        batch_size=batch_size,
        progress_cb=progress_cb,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Train the AutoClipper aesthetic model")
    ap.add_argument("--positives", required=True, type=Path, help="Root folder of positive clips")
    ap.add_argument("--sources", required=True, type=Path, help="Single video file or folder of source videos for negatives")
    ap.add_argument("--output", default=Path("models/aesthetic.pt"), type=Path)
    ap.add_argument("--negatives-dir", type=Path, help="Where to store generated negatives")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    build_and_train(
        args.positives,
        args.sources,
        args.output,
        negatives_dir=args.negatives_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
