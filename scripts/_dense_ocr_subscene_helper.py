#!/usr/bin/env python3
"""Run dense OCR within specified long scenes for the production pipeline.

The helper reads a request JSON from the main environment and returns timestamps
with large OCR-text changes as response JSON. Run it in the dedicated OCR
environment to isolate the heavier dependencies.

This script must run in the `.venv-ocr` environment because it imports
`rapidocr-onnxruntime` and the engine wrappers from `ocr_engine_compare.py`.
The main project pipeline (which lives in `.venv/`) invokes it via subprocess
to keep the heavy OCR dependencies out of the main venv.

Usage:
    .venv-ocr/bin/python scripts/_dense_ocr_subscene_helper.py REQUEST.json RESPONSE.json

REQUEST.json schema:
    {
      "video_path": "data/.../foo.mp4",
      "scenes": [
        {"index": 3, "start_sec": 24.6, "end_sec": 47.0}, ...
      ],
      "fps": 1.0,                  # optional, default 1.0
      "threshold": 0.50,           # optional, default 0.50
      "merge_gap": 2               # optional, default 2
    }

RESPONSE.json schema:
    {
      "boundaries_per_scene": [
        {"index": 3, "boundary_sec": [31.5], "n_frames": 23},
        ...
      ],
      "engine": "rapidocr_gpu",
      "params": {"fps": 1.0, "threshold": 0.50, "merge_gap": 2}
    }

Each `boundary_sec` value is the wall-clock time (in seconds, relative to the
start of the video) at which a telop change was detected within the parent
scene. Downstream code splits the parent scene at these timestamps.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Silence onnxruntime's "GPU device discovery failed" startup warning. We
# already verify CUDAExecutionProvider availability in RapidOCRBackend, so the
# discovery message is just noise on aarch64 headless servers.
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_ocr_sample import (  # noqa: E402
    _pair_distance,
    detect_change_boundaries,
    merge_adjacent_confirmed,
)
from ocr_engine_compare import RapidOCRBackend  # noqa: E402


def extract_frames_in_range(
    mp4: Path, start_sec: float, end_sec: float, fps: float, out_dir: Path,
) -> list[tuple[float, Path]]:
    """Extract frames at `fps` from `start_sec` to `end_sec`.

    Returns [(timestamp_sec_in_video, frame_path), ...]. Frame i in the
    extracted sequence is anchored at start_sec + (i + 0.5) / fps to match the
    mid-interval sampling convention used by the pipeline.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "f_%05d.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", str(start_sec),
         "-to", str(end_sec),
         "-i", str(mp4),
         "-vf", f"fps={fps}",
         "-q:v", "2",
         str(pattern)],
        check=True,
    )
    frames = sorted(out_dir.glob("f_*.jpg"))
    return [(start_sec + (i + 0.5) / fps, p) for i, p in enumerate(frames)]


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: _dense_ocr_subscene_helper.py REQUEST.json RESPONSE.json",
              file=sys.stderr)
        sys.exit(2)
    req_path = Path(sys.argv[1])
    resp_path = Path(sys.argv[2])

    req = json.loads(req_path.read_text())
    mp4 = Path(req["video_path"])
    if not mp4.exists():
        raise SystemExit(f"video_path not found: {mp4}")

    fps = float(req.get("fps", 1.0))
    threshold = float(req.get("threshold", 0.50))
    merge_gap = int(req.get("merge_gap", 2))

    engine = RapidOCRBackend()
    engine.warmup()

    boundaries_per_scene = []
    for s in req["scenes"]:
        idx = s["index"]
        start = float(s["start_sec"])
        end = float(s["end_sec"])
        with tempfile.TemporaryDirectory(prefix=f"dense_ocr_sub_{idx}_") as tmp:
            frames = extract_frames_in_range(mp4, start, end, fps, Path(tmp))
            if len(frames) < 2:
                boundaries_per_scene.append({
                    "index": idx, "boundary_sec": [], "n_frames": len(frames),
                })
                continue

            texts_per_frame: list[tuple[float, list[str]]] = []
            for ts, p in frames:
                r = engine.run(p)
                texts_per_frame.append((ts, r["texts"]))

            distances = [
                _pair_distance(texts_per_frame[i][1], texts_per_frame[i + 1][1])
                for i in range(len(texts_per_frame) - 1)
            ]
            bs = detect_change_boundaries(
                distances, threshold, suppress_isolated_anomaly=False,
            )
            seps = merge_adjacent_confirmed(bs, max_gap=merge_gap)
            # Boundary index i means: change occurred between frame i and i+1.
            # The split timestamp is the wall-clock time of frame i+1's start,
            # i.e. the midpoint between frame i's anchor and frame i+1's anchor.
            boundary_secs = []
            for i in seps:
                t_a = texts_per_frame[i][0]
                t_b = texts_per_frame[i + 1][0]
                boundary_secs.append((t_a + t_b) / 2)
            boundaries_per_scene.append({
                "index": idx,
                "boundary_sec": boundary_secs,
                "n_frames": len(frames),
                "raw_boundaries": [
                    {"i": b["boundary"], "distance": b["distance"]}
                    for b in bs
                ],
            })

    response = {
        "boundaries_per_scene": boundaries_per_scene,
        "engine": "rapidocr_gpu",
        "params": {"fps": fps, "threshold": threshold, "merge_gap": merge_gap},
    }
    resp_path.write_text(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
