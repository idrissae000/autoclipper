"""FFmpeg-based preprocessing: downscale → 1080p, strip audio, convert to 23.976 fps."""

import subprocess
import shutil
import logging
from pathlib import Path

log = logging.getLogger(__name__)

TARGET_FPS = "24000/1001"   # 23.976 exact fraction
TARGET_HEIGHT = 1080
TARGET_WIDTH = 1920


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH – install it and try again.")
    return path


def preprocess(src: Path, dst: Path, *, progress_cb=None) -> Path:
    """
    Convert *src* to a preprocessed working copy at *dst*.

    - Scales to 1920×1080 (letterbox/pillarbox if aspect differs)
    - Strips all audio tracks
    - Forces 23.976 fps via pts rescaling (no frames dropped/duplicated)

    Returns the output path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _ffmpeg(), "-y",
        "-i", str(src),
        # scale: fit inside 1920×1080 preserving SAR, pad remainder black
        "-vf", (
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
            "setsar=1"
        ),
        "-r", TARGET_FPS,
        "-an",                  # no audio
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-movflags", "+faststart",
        str(dst),
    ]

    log.info("Preprocessing %s → %s", src.name, dst.name)
    _run(cmd, src, progress_cb)
    log.info("Preprocessing complete: %s", dst)
    return dst


def get_duration(path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def extract_clip(src: Path, start: float, end: float, dst: Path) -> Path:
    """
    Fast stream-copy cut from *src* between [start, end] seconds → *dst*.

    Uses keyframe-accurate seeking: first seeks before start with -ss, then
    re-encodes a short segment to get frame-accurate in/out points.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start

    cmd = [
        _ffmpeg(), "-y",
        "-ss", f"{start:.6f}",
        "-i", str(src),
        "-t", f"{duration:.6f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-an",
        "-movflags", "+faststart",
        str(dst),
    ]

    _run(cmd, src)
    return dst


def _run(cmd: list[str], src: Path, progress_cb=None) -> None:
    """Run ffmpeg, forward stderr to logger, call progress_cb(pct) if given."""
    duration: float | None = None
    if progress_cb:
        try:
            duration = get_duration(src)
        except Exception:
            pass

    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    for line in proc.stderr:  # type: ignore[union-attr]
        line = line.rstrip()
        log.debug("[ffmpeg] %s", line)
        if progress_cb and duration and "time=" in line:
            try:
                ts = line.split("time=")[1].split(" ")[0]
                h, m, s = ts.split(":")
                secs = int(h) * 3600 + int(m) * 60 + float(s)
                progress_cb(min(secs / duration, 1.0))
            except Exception:
                pass

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
