#!/usr/bin/env python3
"""Shared utilities for detecting on-screen text changes with RapidOCR."""

from __future__ import annotations

import subprocess
import unicodedata
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein


DEFAULTS = {
    "fps": 1.0,
    "threshold": 0.50,
    "merge_gap": 2,
    "suppress_isolated_anomaly": False,
    "anomaly_max_distance": 0.70,
    "anomaly_max_diff": 0.15,
    "gray_zone": (0.45, 0.60),
}


def _norm(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value or "").split()).lower()


def _join_norm(texts: list[str]) -> str:
    return " ".join(_norm(text) for text in texts if _norm(text))


def _pair_distance(previous: list[str], current: list[str]) -> float:
    left, right = _join_norm(previous), _join_norm(current)
    if not left and not right:
        return 0.0
    return Levenshtein.normalized_distance(left, right)


def _ffprobe_duration(video: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        text=True,
    ).strip()
    return float(output)


def _extract_frames(video: Path, fps: float, output_dir: Path) -> list[tuple[float, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "f_%05d.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            str(pattern),
        ],
        check=True,
    )
    frames = sorted(output_dir.glob("f_*.jpg"))
    return [((index + 0.5) / fps, path) for index, path in enumerate(frames)]


def detect_change_boundaries(
    distances: list[float],
    threshold: float,
    suppress_isolated_anomaly: bool,
) -> list[dict[str, Any]]:
    """Create boundary candidates from adjacent-frame OCR distances."""

    high = [distance > threshold for distance in distances]
    suppressed: set[int] = set()
    if suppress_isolated_anomaly:
        maximum = DEFAULTS["anomaly_max_distance"]
        maximum_difference = DEFAULTS["anomaly_max_diff"]
        for index in range(1, len(distances)):
            if not (high[index - 1] and high[index]):
                continue
            left_isolated = index - 2 < 0 or not high[index - 2]
            right_isolated = index + 1 >= len(distances) or not high[index + 1]
            moderate = distances[index - 1] <= maximum and distances[index] <= maximum
            similar = abs(distances[index - 1] - distances[index]) <= maximum_difference
            if left_isolated and right_isolated and moderate and similar:
                suppressed.update((index - 1, index))

    return [
        {
            "boundary": index,
            "distance": distance,
            "confirmed": index not in suppressed,
            "suppressed_as_anomaly": index in suppressed,
            "kind": "gray" if distance < DEFAULTS["gray_zone"][1] else "high",
        }
        for index, distance in enumerate(distances)
        if distance > threshold
    ]


def merge_adjacent_confirmed(
    boundaries: list[dict[str, Any]], max_gap: int = 2
) -> list[int]:
    """Merge nearby confirmed candidates and return each cluster's first index."""

    confirmed = sorted(row["boundary"] for row in boundaries if row["confirmed"])
    if not confirmed:
        return []
    clusters = [[confirmed[0]]]
    for index in confirmed[1:]:
        if index - clusters[-1][-1] <= max_gap:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    return [cluster[0] for cluster in clusters]
