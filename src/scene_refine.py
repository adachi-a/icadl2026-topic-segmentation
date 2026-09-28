"""Refine long scene ranges before frame annotation.

Long ranges are rechecked with ContentDetector, adjusted to nearby audio
transitions, and normalized to avoid extremely short or long ranges. External
video operations are isolated from the deterministic range transformations."""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

DEFAULT_LONG_SCENE_SEC = 20.0
DEFAULT_CONTENT_THRESHOLD = 27.0
DEFAULT_CONTENT_MIN_SCENE_LEN_FRAMES = 30
DEFAULT_SILENCE_NOISE_DB = -30.0
DEFAULT_SILENCE_MIN_SEC = 0.35
DEFAULT_SNAP_TOLERANCE_SEC = 1.0
DEFAULT_MIN_SCENE_SEC = 2.0
DEFAULT_MAX_SCENE_SEC = 30.0
DEFAULT_FORCE_SPLIT_TARGET_SEC = 20.0
DEFAULT_EDGE_MARGIN_SEC = 0.5

Range = tuple[float, float]


def snap_cuts_to_silence_ends(
    cuts: list[float],
    silence_ends: list[float],
    tolerance_sec: float = DEFAULT_SNAP_TOLERANCE_SEC,
) -> tuple[list[float], int]:
    """Move each cut to the nearest silence_end (= speech onset) within tolerance.

    Returns the sorted, deduplicated cuts and the number of cuts that moved
    (cuts that collapse onto the same onset are all counted as snapped).
    """
    if not silence_ends:
        return sorted(set(cuts)), 0
    snapped: list[float] = []
    n_snapped = 0
    for cut in cuts:
        nearest = min(silence_ends, key=lambda end: abs(end - cut))
        if abs(nearest - cut) <= tolerance_sec and nearest != cut:
            snapped.append(nearest)
            n_snapped += 1
        else:
            snapped.append(cut)
    return sorted(set(snapped)), n_snapped


def insert_cuts_into_ranges(
    ranges: list[Range],
    cuts: list[float],
    edge_margin_sec: float = DEFAULT_EDGE_MARGIN_SEC,
) -> tuple[list[Range], int]:
    """Split ranges at the given cuts; cuts within edge_margin of a range
    boundary (or outside every range) are dropped."""
    out: list[Range] = []
    for start, end in ranges:
        inner = sorted({
            cut for cut in cuts
            if start + edge_margin_sec < cut < end - edge_margin_sec
        })
        cursor = start
        for cut in inner:
            out.append((cursor, cut))
            cursor = cut
        out.append((cursor, end))
    return out, len(out) - len(ranges)


def merge_short_ranges(
    ranges: list[Range],
    min_len_sec: float = DEFAULT_MIN_SCENE_SEC,
) -> tuple[list[Range], int]:
    """Merge ranges shorter than min_len_sec into their left neighbour
    (or into the right neighbour when there is nothing on the left)."""
    out: list[Range] = []
    pending_start: float | None = None
    for start, end in ranges:
        if pending_start is not None:
            start = pending_start
            pending_start = None
        if end - start < min_len_sec:
            if out:
                out[-1] = (out[-1][0], end)
            else:
                pending_start = start
            continue
        out.append((start, end))
    if pending_start is not None:
        out.append((pending_start, ranges[-1][1]))
    return out, len(ranges) - len(out)


def force_split_long_ranges(
    ranges: list[Range],
    max_len_sec: float = DEFAULT_MAX_SCENE_SEC,
    target_len_sec: float = DEFAULT_FORCE_SPLIT_TARGET_SEC,
) -> tuple[list[Range], int]:
    """Split every range longer than max_len_sec into equal parts of roughly
    target_len_sec (insurance so the VLM sees inside visually uniform spans)."""
    out: list[Range] = []
    for start, end in ranges:
        duration = end - start
        if duration <= max_len_sec:
            out.append((start, end))
            continue
        n_parts = math.ceil(duration / target_len_sec)
        points = [start + duration * i / n_parts for i in range(1, n_parts)]
        cursor = start
        for point in points:
            out.append((cursor, point))
            cursor = point
        out.append((cursor, end))
    return out, len(out) - len(ranges)


def _media_start_time_sec(media_path: str | Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=start_time",
         "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def detect_silence_ends(
    media_path: str | Path,
    noise_db: float = DEFAULT_SILENCE_NOISE_DB,
    min_silence_sec: float = DEFAULT_SILENCE_MIN_SEC,
) -> list[float]:
    """Speech onsets via ffmpeg silencedetect. Prefer the run's media/audio.wav
    (program-relative time base); when reading a TS directly, the container
    start_time is subtracted to stay program-relative."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-vn",
        "-i", str(media_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([0-9.]+)", proc.stderr)]
    offset = _media_start_time_sec(media_path)
    if offset > 1.0:
        ends = [end - offset for end in ends if end >= offset]
    return sorted(set(ends))


def detect_second_pass_cuts(
    video_path: str | Path,
    long_ranges: list[Range],
    threshold: float = DEFAULT_CONTENT_THRESHOLD,
    min_scene_len_frames: int = DEFAULT_CONTENT_MIN_SCENE_LEN_FRAMES,
) -> list[float]:
    """ContentDetector cuts falling inside the given long scenes.

    Runs one whole-video pass (cheaper than seeking per range) and keeps only
    cuts inside a long range; the hash detector's own boundaries are untouched.
    """
    if not long_ranges:
        return []
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    from src.extract_frames import (
        _ffprobe_duration,
        _filter_libav_stderr,
        _scene_list_to_relative_seconds,
    )

    duration = _ffprobe_duration(video_path)
    with _filter_libav_stderr():
        video = open_video(str(video_path), backend="pyav")
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
        )
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
    if not scene_list:
        return []
    scenes_sec, _offset = _scene_list_to_relative_seconds(scene_list, duration)
    cuts = [end for _start, end in scenes_sec[:-1]]
    return sorted(
        cut for cut in cuts
        if any(start < cut < end for start, end in long_ranges)
    )


def refine_scene_ranges(
    video_path: str | Path,
    scene_ranges: list[Range],
    *,
    silence_media_path: str | Path | None = None,
    long_scene_sec: float = DEFAULT_LONG_SCENE_SEC,
    content_threshold: float = DEFAULT_CONTENT_THRESHOLD,
    content_min_scene_len_frames: int = DEFAULT_CONTENT_MIN_SCENE_LEN_FRAMES,
    silence_noise_db: float = DEFAULT_SILENCE_NOISE_DB,
    silence_min_sec: float = DEFAULT_SILENCE_MIN_SEC,
    snap_tolerance_sec: float = DEFAULT_SNAP_TOLERANCE_SEC,
    min_scene_sec: float = DEFAULT_MIN_SCENE_SEC,
    max_scene_sec: float = DEFAULT_MAX_SCENE_SEC,
    force_split_target_sec: float = DEFAULT_FORCE_SPLIT_TARGET_SEC,
    second_pass_cuts: list[float] | None = None,
    silence_ends: list[float] | None = None,
) -> tuple[list[Range], dict]:
    """Apply the full refine chain to program-relative scene ranges.

    `second_pass_cuts` / `silence_ends` are injectable for tests; when None
    they are detected from the video / audio.
    """
    long_ranges = [r for r in scene_ranges if r[1] - r[0] > long_scene_sec]
    if second_pass_cuts is None:
        second_pass_cuts = detect_second_pass_cuts(
            video_path, long_ranges,
            threshold=content_threshold,
            min_scene_len_frames=content_min_scene_len_frames,
        )
    silence_source = str(silence_media_path or video_path)
    if silence_ends is None:
        silence_ends = detect_silence_ends(
            silence_media_path or video_path,
            noise_db=silence_noise_db,
            min_silence_sec=silence_min_sec,
        )

    snapped_cuts, n_snapped = snap_cuts_to_silence_ends(
        second_pass_cuts, silence_ends, tolerance_sec=snap_tolerance_sec
    )
    ranges, n_inserted = insert_cuts_into_ranges(scene_ranges, snapped_cuts)
    ranges, n_merged = merge_short_ranges(ranges, min_len_sec=min_scene_sec)
    ranges, n_forced = force_split_long_ranges(
        ranges, max_len_sec=max_scene_sec, target_len_sec=force_split_target_sec
    )

    # Round via the shared boundary list so contiguity survives rounding.
    bounds = [ranges[0][0]] + [end for _start, end in ranges]
    bounds = [round(b, 2) for b in bounds]
    rounded = [
        (bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i + 1] > bounds[i]
    ]

    stats = {
        "enabled": True,
        "params": {
            "long_scene_sec": long_scene_sec,
            "content_threshold": content_threshold,
            "content_min_scene_len_frames": content_min_scene_len_frames,
            "silence_noise_db": silence_noise_db,
            "silence_min_sec": silence_min_sec,
            "snap_tolerance_sec": snap_tolerance_sec,
            "min_scene_sec": min_scene_sec,
            "max_scene_sec": max_scene_sec,
            "force_split_target_sec": force_split_target_sec,
        },
        "silence_source": silence_source,
        "n_silence_ends": len(silence_ends),
        "n_scenes_in": len(scene_ranges),
        "n_long_scenes": len(long_ranges),
        "n_second_pass_cuts": len(second_pass_cuts),
        "n_cuts_snapped": n_snapped,
        "n_cuts_inserted": n_inserted,
        "n_merged": n_merged,
        "n_forced_split_cuts": n_forced,
        "n_scenes_out": len(rounded),
    }
    return rounded, stats
