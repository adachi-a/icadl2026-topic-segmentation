#!/usr/bin/env python3
"""Extract telop change-point candidates from an arbitrary broadcast video.

Run with `.venv-ocr/bin/python` and provide `--video` and `--out`. The output
contains the video duration and the detected boundary times."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dense_ocr_sample import (  # noqa: E402
    DEFAULTS,
    _ffprobe_duration,
    _extract_frames,
    _pair_distance,
    detect_change_boundaries,
    merge_adjacent_confirmed,
)


def run(
    video: Path,
    out_path: Path,
    fps: float = DEFAULTS["fps"],
    threshold: float = DEFAULTS["threshold"],
    merge_gap: int = DEFAULTS["merge_gap"],
    suppress_isolated_anomaly: bool = DEFAULTS["suppress_isolated_anomaly"],
    keep_frames_dir: Path | None = None,
) -> dict:
    from ocr_engine_compare import RapidOCRBackend

    duration = _ffprobe_duration(video)
    print(f"[telop_ocr] {video.name}: duration={duration:.1f}s, "
          f"fps={fps}, threshold={threshold}, merge_gap={merge_gap}")

    if keep_frames_dir is not None:
        frames_dir = keep_frames_dir
        frames_dir.mkdir(parents=True, exist_ok=True)
        owns_dir = False
    else:
        frames_dir = Path(tempfile.mkdtemp(prefix=f"telop_ocr_{video.stem[:20]}_"))
        owns_dir = True

    try:
        t0 = time.time()
        frames = _extract_frames(video, fps, frames_dir)
        print(f"[telop_ocr] {len(frames)} frames extracted in {time.time()-t0:.1f}s")

        engine = RapidOCRBackend()
        engine.warmup()

        per_frame = []
        t1 = time.time()
        for idx, (ts, path) in enumerate(frames):
            r = engine.run(path)
            per_frame.append({
                "frame_index": idx,
                "timestamp_sec": ts,
                "texts": r["texts"],
            })
            if (idx + 1) % 200 == 0:
                print(f"[telop_ocr] OCR progress: {idx+1}/{len(frames)} "
                      f"({(idx+1)/(time.time()-t1):.1f} fps)")
        ocr_elapsed = time.time() - t1
        print(f"[telop_ocr] OCR done: {len(per_frame)} frames in {ocr_elapsed:.1f}s "
              f"({len(per_frame)/ocr_elapsed:.1f} fps)")

        # Pairwise distance series
        distances = []
        for i in range(len(per_frame) - 1):
            d = _pair_distance(per_frame[i]["texts"], per_frame[i + 1]["texts"])
            distances.append(d)

        boundaries = detect_change_boundaries(
            distances, threshold, suppress_isolated_anomaly,
        )
        separators = merge_adjacent_confirmed(boundaries, max_gap=merge_gap)

        # Convert separator index → seconds (start of frame i+1, the new event's start)
        # _extract_frames assigns ts = (i + 0.5) / fps, so frame[i+1].ts = (i + 1.5) / fps
        # We use the integer second (i + 1) / fps as boundary_sec
        boundaries_sec = sorted({round((i + 1) / fps, 2) for i in separators})

        result = {
            "video": str(video),
            "duration_sec": duration,
            "fps": fps,
            "params": {
                "threshold": threshold,
                "merge_gap": merge_gap,
                "suppress_isolated_anomaly": suppress_isolated_anomaly,
            },
            "stats": {
                "n_frames": len(per_frame),
                "n_distances": len(distances),
                "n_change_candidates": sum(1 for b in boundaries if b["confirmed"]),
                "n_separators": len(separators),
                "n_boundaries": len(boundaries_sec),
                "ocr_elapsed_sec": round(ocr_elapsed, 1),
            },
            "boundaries_sec": boundaries_sec,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[telop_ocr] wrote {out_path}: {len(boundaries_sec)} boundaries")
        return result

    finally:
        if owns_dir:
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(
        description="1fps rapidocr telop boundary extractor for broadcast videos"
    )
    ap.add_argument("--video", type=Path, required=True, help="Input MP4")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path")
    ap.add_argument("--fps", type=float, default=DEFAULTS["fps"])
    ap.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    ap.add_argument("--merge-gap", type=int, default=DEFAULTS["merge_gap"])
    ap.add_argument("--suppress-anomaly", action="store_true")
    ap.add_argument(
        "--keep-frames-dir", type=Path, default=None,
        help="Persist extracted frames here (otherwise tempdir cleared on exit)",
    )
    args = ap.parse_args()

    run(
        video=args.video,
        out_path=args.out,
        fps=args.fps,
        threshold=args.threshold,
        merge_gap=args.merge_gap,
        suppress_isolated_anomaly=args.suppress_anomaly,
        keep_frames_dir=args.keep_frames_dir,
    )


if __name__ == "__main__":
    main()
