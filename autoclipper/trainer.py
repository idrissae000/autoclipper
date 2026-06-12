"""
Training entry point: scan positive clip library, generate negatives, train model.

Usage (CLI):
  python -m autoclipper.trainer \
      --positives /clips/show_a /clips/show_b \
      --sources   /raw/movie1.mp4 /raw/season2 \
      --output    models/aesthetic.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from autoclipper.constants import VIDEO_EXTENSIONS, scan_for_videos

log = logging.getLogger(__name__)


def collect_positives(roots: list[Path]) -> list[Path]:
    """Recursively collect all video clips under one or more *roots*."""
    clips: list[Path] = []
    for root in roots:
        clips.extend(scan_for_videos(root))
    result = sorted(set(clips))
    log.info("Found %d positive clips across %d director%s",
             len(result), len(roots), "y" if len(roots) == 1 else "ies")
    return result


def resolve_source_videos(sources: list[Path]) -> list[Path]:
    """
    Accept any mix of individual video files and folders.

    Files are validated and used directly.
    Folders are scanned recursively for all supported video formats.
    Returns a deduplicated, sorted list.
    """
    videos: list[Path] = []
    for src in sources:
        if src.is_file():
            if src.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"Not a recognised video file: {src}")
            videos.append(src)
        else:
            videos.extend(scan_for_videos(src))
    return sorted(set(videos))


def build_and_train(
    positives_dirs: list[Path],
    source_videos: list[Path],
    output_model: Path,
    *,
    negatives_dir: Path | None = None,
    epochs: int = 20,
    batch_size: int = 16,
    progress_cb=None,
):
    from autoclipper.clip_scorer import generate_negatives, train

    positives = collect_positives(positives_dirs)
    if not positives:
        raise ValueError(f"No positive clips found in: {positives_dirs}")

    neg_dir = negatives_dir or output_model.parent / "negatives"
    source_paths = resolve_source_videos(source_videos)

    if not source_paths:
        raise ValueError(f"No source videos found in: {source_videos}")

    log.info("Generating negative examples from %d source videos…", len(source_paths))
    negatives = generate_negatives(
        source_paths,
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
    ap.add_argument(
        "--positives", required=True, nargs="+", type=Path,
        help="One or more folders of positive clips",
    )
    ap.add_argument(
        "--sources", required=True, nargs="+", type=Path,
        help="One or more video files or folders for negative generation",
    )
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
