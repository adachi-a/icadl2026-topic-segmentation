"""Create scene ranges and representative frames from video.

The module aligns fixed-interval samples, PySceneDetect cuts, OCR text-change
points, and supplied boundaries to one program-relative timeline. Scene-based
sampling avoids the instant immediately after a cut and keeps frame images for
downstream processing. Dense OCR runs in a helper subprocess so its heavier
dependencies stay outside this module.
"""

import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import cv2

# Try PyAV's logging knob first (does cut some warnings on some builds).
try:
    import av
    av.logging.set_level(av.logging.ERROR)
except Exception:
    pass


# libav's swscale subsystem writes "[swscaler @ 0x…] No accelerated colorspace
# conversion found from yuv420p to bgr24" directly to fd 2 from C, bypassing
# Python-side logging. On aarch64 this fires once per frame during
# PySceneDetect's HashDetector run, which can produce thousands of lines and
# drown out the actual progress output.
#
# Solution: redirect fd 2 through a pipe during the noisy section, run a
# filter thread that drops only lines matching the swscaler pattern, and
# forwards everything else to the real stderr so genuine errors stay visible.
_LIBAV_NOISE_RE = re.compile(rb"\[swscaler @ 0x[0-9a-f]+\]")


@contextlib.contextmanager
def _filter_libav_stderr():
    """Drop libav swscaler lines from fd 2 for the duration of the with-block.

    This is a context-managed file-descriptor swap. Concurrent threads writing
    to stderr during the block are also affected, so keep the block as small
    as possible (around the libav-using call).
    """
    if not sys.stderr.isatty() and not getattr(sys.stderr, "buffer", None):
        # Some pytest captures replace stderr; bail out and skip the redirect.
        yield
        return
    real_fd = os.dup(2)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)
    stop = threading.Event()

    def _pump():
        buf = b""
        try:
            while not stop.is_set():
                try:
                    chunk = os.read(r_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not _LIBAV_NOISE_RE.search(line):
                        os.write(real_fd, line + b"\n")
            # Final flush of any unterminated line
            if buf and not _LIBAV_NOISE_RE.search(buf):
                os.write(real_fd, buf)
        except Exception:
            pass

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    try:
        yield
    finally:
        # Restore fd 2 first so subsequent stderr writes go straight to real_fd
        os.dup2(real_fd, 2)
        os.close(real_fd)
        stop.set()
        # Closing the read end signals the pump to exit
        try:
            os.close(r_fd)
        except OSError:
            pass
        thread.join(timeout=2)

# Path to the OCR-only venv used by `_dense_ocr_subscene_helper.py`. The helper
# imports rapidocr-onnxruntime / onnxruntime-gpu which are intentionally absent
# from the main project venv (see requirements-ocr.txt).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
OCR_VENV_PYTHON = _PROJECT_ROOT / ".venv-ocr" / "bin" / "python"
OCR_HELPER_SCRIPT = _PROJECT_ROOT / "scripts" / "_dense_ocr_subscene_helper.py"


# ============================================================
# Face detection
# ============================================================

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)


def detect_faces(image_path: str | Path) -> list[dict]:
    """Detect faces and return their positions and sizes.

    Returns:
        [{"x": int, "y": int, "w": int, "h": int}, ...]
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            for x, y, w, h in detections]


def extract_frames(
    video_path: str | Path,
    output_dir: Path,
    interval_sec: float = 5.0,
    max_dimension: int = 1280,
) -> list[dict]:
    """Extract frames from a video at a fixed interval.

    Args:
        video_path: Input video path.
        output_dir: Output directory for frame images.
        interval_sec: Sampling interval in seconds.
        max_dimension: Maximum size of the longer edge.

    Returns:
        Metadata for the extracted frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract fixed-interval frames with FFmpeg. The fps filter samples once
    # every N seconds, and scale limits the longer edge to max_dimension.
    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps=1/{interval_sec},scale='if(gt(iw,ih),{max_dimension},-2)':'if(gt(ih,iw),{max_dimension},-2)'",
            "-q:v", "2",  # JPEG quality
            str(output_dir / "frame_%04d.jpg"),
            "-y",  # Overwrite existing output.
        ],
        check=True,
        capture_output=True,
    )

    # Build the extracted-frame records.
    frames = []
    for i, path in enumerate(sorted(output_dir.glob("frame_*.jpg"))):
        frames.append({
            "frame_index": i,
            "path": str(path),
            "timestamp_sec": i * interval_sec,
        })

    return frames


def _call_dense_ocr_subscene_helper(
    video_path: Path,
    long_scenes: list[dict],
    *,
    fps: float = 1.0,
    threshold: float = 0.50,
    merge_gap: int = 2,
) -> dict[int, list[float]]:
    """Invoke the .venv-ocr-based dense OCR helper to find sub-scene telop
    boundaries inside the given long scenes.

    `long_scenes` is a list of {"index": int, "start_sec": float, "end_sec": float}.

    Returns a dict mapping scene_index -> [boundary_sec, ...]. Boundary
    timestamps are wall-clock times within the original video.
    """
    if not OCR_VENV_PYTHON.exists():
        raise RuntimeError(
            f"OCR venv python not found at {OCR_VENV_PYTHON}. "
            "Set up with: uv venv .venv-ocr --python 3.12 && "
            "uv pip install --python .venv-ocr/bin/python -r requirements-ocr.txt"
        )
    if not OCR_HELPER_SCRIPT.exists():
        raise RuntimeError(f"OCR helper not found at {OCR_HELPER_SCRIPT}")

    request = {
        "video_path": str(video_path),
        "scenes": long_scenes,
        "fps": fps,
        "threshold": threshold,
        "merge_gap": merge_gap,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="dense_ocr_req_", delete=False,
    ) as req_file:
        json.dump(request, req_file)
        req_path = Path(req_file.name)
    resp_path = req_path.with_name(req_path.stem + "_resp.json")

    try:
        # Capture the helper's stderr so its onnxruntime-init warnings
        # ("GPU device discovery failed", "[W:onnxruntime:Default ...]") and
        # ffmpeg "[swscaler @ 0x...]" lines do not pollute the parent's stream.
        # Forward stderr only when the helper itself fails — keeping real
        # diagnostics, dropping cosmetic startup noise.
        proc = subprocess.run(
            [str(OCR_VENV_PYTHON), str(OCR_HELPER_SCRIPT),
             str(req_path), str(resp_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            if proc.stderr:
                sys.stderr.write(proc.stderr)
            raise subprocess.CalledProcessError(
                proc.returncode, proc.args, proc.stdout, proc.stderr,
            )
        response = json.loads(resp_path.read_text())
    finally:
        for p in (req_path, resp_path):
            try:
                p.unlink()
            except OSError:
                pass

    return {
        item["index"]: list(item.get("boundary_sec") or [])
        for item in response.get("boundaries_per_scene", [])
    }


def _normalize_boundaries_sec(boundaries_sec: list[float] | None) -> list[float]:
    if boundaries_sec is None:
        return []
    return sorted({round(float(value), 2) for value in boundaries_sec})


def _boundaries_by_scene_from_global(
    boundaries_sec: list[float] | None,
    scenes: list[dict],
) -> dict[int, list[float]]:
    """Assign full-video telop OCR boundaries to parent scene ranges."""
    global_boundaries = _normalize_boundaries_sec(boundaries_sec)
    out: dict[int, list[float]] = {}
    for scene in scenes:
        idx = int(scene["index"])
        start = float(scene["start_sec"])
        end = float(scene["end_sec"])
        out[idx] = [
            boundary for boundary in global_boundaries
            if start < boundary < end
        ]
    return out


def _split_scenes_at_boundaries(
    scene_list: list[tuple],
    boundaries_by_scene: dict[int, list[float]],
    video_frame_rate: float,
) -> list[tuple]:
    """Insert sub-scene boundaries into the original scene list.

    Each entry of `boundaries_by_scene` maps the index in the original
    scene_list to a list of timestamps (seconds) that should split that scene.

    Returns a new scene_list where each long-shot may have been broken into
    multiple consecutive (start_tc, end_tc) tuples.
    """
    from scenedetect import FrameTimecode

    new_list = []
    for idx, (start_tc, end_tc) in enumerate(scene_list):
        bounds = sorted(b for b in boundaries_by_scene.get(idx, [])
                        if start_tc.get_seconds() < b < end_tc.get_seconds())
        if not bounds:
            new_list.append((start_tc, end_tc))
            continue
        cur = start_tc
        for b in bounds:
            b_tc = FrameTimecode(timecode=b, fps=video_frame_rate)
            new_list.append((cur, b_tc))
            cur = b_tc
        new_list.append((cur, end_tc))
    return new_list


def _ffprobe_duration(video_path: Path) -> float:
    """Return the full video duration in seconds using ffprobe."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(proc.stdout.strip())


def _scene_list_to_relative_seconds(scene_list: list[tuple], duration: float) -> tuple[list[tuple[float, float]], float]:
    """Convert PySceneDetect FrameTimecode ranges to program-relative seconds.

    Broadcast TS files often carry a large PTS start offset. PySceneDetect may
    expose scene times in that absolute timeline, e.g. 89503s for the first
    video frame of a 3500s recording. Downstream metadata must use seconds from
    program start, so subtract the first scene start and clamp to duration.
    """
    raw: list[tuple[float, float]] = [
        (float(start.get_seconds()), float(end.get_seconds()))
        for start, end in scene_list
    ]
    if not raw:
        return ([(0.0, round(duration, 2))] if duration > 0 else []), 0.0

    offset = raw[0][0]
    normalized: list[tuple[float, float]] = []
    for start, end in raw:
        rel_start = max(0.0, start - offset)
        rel_end = max(0.0, end - offset)
        rel_start = min(rel_start, duration)
        rel_end = min(rel_end, duration)
        if rel_end > rel_start:
            normalized.append((round(rel_start, 2), round(rel_end, 2)))

    if not normalized and duration > 0:
        normalized = [(0.0, round(duration, 2))]
    return normalized, round(offset, 6)


def detect_scenes(
    video_path: str | Path,
    strategy: str = "option_d",
    detector: str = "hash",
    threshold: float | None = None,
    min_scene_len: int = 90,
    long_shot_threshold_sec: float = 10.0,
    ocr_subscene_threshold: float = 0.50,
    ocr_fps: float = 1.0,
    ocr_merge_gap: int = 2,
    ocr_boundaries_sec: list[float] | None = None,
) -> tuple[list[tuple[float, float]], list[int], dict]:
    """Return scene boundaries for the selected splitting strategy.

    Args:
        strategy:
            - ``pysd_only``: PySceneDetect HashDetector without OCR.
            - ``option_d``: PySceneDetect plus OCR subdivision of scenes at
              least ``long_shot_threshold_sec`` long.
            - ``ocr_only``: OCR change-point detection over the full video.
        ocr_boundaries_sec:
            Precomputed full-video on-screen text boundaries. When supplied,
            scene detection reuses them instead of rerunning OCR.

    Returns:
        (scenes, subscene_origin, stats):
        - scenes: list of (start_sec, end_sec)
        - subscene_origin: Parent-scene index for each resulting range.
        - stats: {"n_pysd_scenes": int, "n_long_scenes": int, "n_subscenes_added": int,
                  "duration_sec": float, "strategy": str}
    """
    from scenedetect import open_video, SceneManager, FrameTimecode
    from scenedetect.detectors import ContentDetector, AdaptiveDetector, HashDetector

    if strategy not in ("pysd_only", "option_d", "ocr_only"):
        raise ValueError(f"unknown strategy: {strategy}")

    video_path = Path(video_path)
    duration = _ffprobe_duration(video_path)
    stats = {
        "strategy": strategy,
        "duration_sec": round(duration, 2),
        "n_pysd_scenes": 0,
        "n_long_scenes": 0,
        "n_subscenes_added": 0,
        "ocr_boundary_source": "none",
        "n_ocr_boundaries_provided": len(_normalize_boundaries_sec(ocr_boundaries_sec)),
    }

    # OCR-only change-point detection over the full video.
    if strategy == "ocr_only":
        if ocr_boundaries_sec is not None:
            bounds = [
                boundary for boundary in _normalize_boundaries_sec(ocr_boundaries_sec)
                if 0.0 < boundary < duration
            ]
            stats["ocr_boundary_source"] = "provided_full_video"
        else:
            whole_video = [{"index": 0, "start_sec": 0.0, "end_sec": duration}]
            boundaries_by_scene = _call_dense_ocr_subscene_helper(
                video_path, whole_video,
                fps=ocr_fps, threshold=ocr_subscene_threshold, merge_gap=ocr_merge_gap,
            )
            bounds = sorted(b for b in boundaries_by_scene.get(0, []) if 0.0 < b < duration)
            stats["ocr_boundary_source"] = "dense_ocr_helper"
        cuts = [0.0] + bounds + [duration]
        scenes = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        stats["n_subscenes_added"] = len(bounds)
        return scenes, list(range(len(scenes))), stats

    # ---- S-A / S-B: PySceneDetect HashDetector ----
    if detector == "hash":
        det = HashDetector(
            threshold=threshold if threshold is not None else 0.42,
            min_scene_len=min_scene_len,
        )
    elif detector == "adaptive":
        det = AdaptiveDetector(
            adaptive_threshold=threshold if threshold is not None else 3.0,
            min_scene_len=min_scene_len,
        )
    else:
        det = ContentDetector(
            threshold=threshold if threshold is not None else 27.0,
            min_scene_len=min_scene_len,
        )

    with _filter_libav_stderr():
        video = open_video(str(video_path), backend="pyav")
        scene_manager = SceneManager()
        scene_manager.add_detector(det)
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

    if not scene_list:
        stats["n_pysd_scenes"] = 1
        stats["scene_time_offset_sec"] = 0.0
        scenes = [(0.0, round(duration, 2))] if duration > 0 else []
        return scenes, list(range(len(scenes))), stats

    stats["n_pysd_scenes"] = len(scene_list)
    pysd_scenes_sec, scene_time_offset = _scene_list_to_relative_seconds(scene_list, duration)
    stats["scene_time_offset_sec"] = scene_time_offset
    subscene_origin = list(range(len(pysd_scenes_sec)))

    # Return without OCR subdivision.
    if strategy == "pysd_only":
        return pysd_scenes_sec, subscene_origin, stats

    # Subdivide long scenes using OCR change points.
    long_scenes = []
    for idx, (start_sec, end_sec) in enumerate(pysd_scenes_sec):
        if end_sec - start_sec >= long_shot_threshold_sec:
            long_scenes.append({
                "index": idx,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
            })
    stats["n_long_scenes"] = len(long_scenes)

    if not long_scenes:
        return pysd_scenes_sec, subscene_origin, stats

    if ocr_boundaries_sec is not None:
        boundaries_by_scene = _boundaries_by_scene_from_global(
            ocr_boundaries_sec, long_scenes,
        )
        stats["ocr_boundary_source"] = "provided_full_video"
    else:
        boundaries_by_scene = _call_dense_ocr_subscene_helper(
            video_path, long_scenes,
            fps=ocr_fps, threshold=ocr_subscene_threshold, merge_gap=ocr_merge_gap,
        )
        stats["ocr_boundary_source"] = "dense_ocr_helper"

    new_scene_list: list[tuple[float, float]] = []
    new_origin: list[int] = []
    n_subscenes_added = 0
    for idx, (start_sec, end_sec) in enumerate(pysd_scenes_sec):
        bounds = sorted(
            b for b in boundaries_by_scene.get(idx, [])
            if start_sec < b < end_sec
        )
        if not bounds:
            new_scene_list.append((float(start_sec), float(end_sec)))
            new_origin.append(idx)
            continue
        cur_sec = float(start_sec)
        for b in bounds:
            new_scene_list.append((cur_sec, float(b)))
            new_origin.append(idx)
            cur_sec = float(b)
            n_subscenes_added += 1
        new_scene_list.append((cur_sec, float(end_sec)))
        new_origin.append(idx)

    stats["n_subscenes_added"] = n_subscenes_added
    return new_scene_list, new_origin, stats


def extract_frames_by_scene(
    video_path: str | Path,
    output_dir: Path,
    detector: str = "hash",
    threshold: float | None = None,
    min_scene_len: int = 90,
    max_dimension: int = 1280,
    offset_sec: float = 0.5,
    use_dense_ocr_for_long_shots: bool = False,
    long_shot_threshold_sec: float = 10.0,
    ocr_subscene_threshold: float = 0.50,
    ocr_fps: float = 1.0,
    ocr_merge_gap: int = 2,
    ocr_boundaries_sec: list[float] | None = None,
) -> list[dict]:
    """Detect scene changes and extract one representative frame per scene.

    Args:
        video_path: Input video path.
        output_dir: Output directory for frame images.
        detector: Detection algorithm: ``content``, ``adaptive``, or ``hash``.
        threshold: Detection threshold, or the detector default when None.
            Lower values increase sensitivity for content and adaptive
            detectors but decrease sensitivity for the hash detector.
        min_scene_len: Minimum scene length in frames.
        max_dimension: Maximum size of the longer image edge.
        use_dense_ocr_for_long_shots: If true, sample long scenes with dense
            OCR and split them at on-screen text changes. Requires
            ``.venv-ocr/`` and ``scripts/_dense_ocr_subscene_helper.py``.
        long_shot_threshold_sec: Minimum duration of a long scene.
        ocr_subscene_threshold: OCR edit-distance threshold.
        ocr_fps: Dense OCR sampling rate.
        ocr_merge_gap: Maximum gap used to merge adjacent OCR boundaries.
        ocr_boundaries_sec: Precomputed full-video OCR boundaries. When
            supplied, only boundaries inside long scenes are used.
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector, AdaptiveDetector, HashDetector

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if detector == "hash":
        det = HashDetector(
            threshold=threshold if threshold is not None else 0.42,
            min_scene_len=min_scene_len,
        )
    elif detector == "adaptive":
        det = AdaptiveDetector(
            adaptive_threshold=threshold if threshold is not None else 3.0,
            min_scene_len=min_scene_len,
        )
    else:
        det = ContentDetector(
            threshold=threshold if threshold is not None else 27.0,
            min_scene_len=min_scene_len,
        )

    with _filter_libav_stderr():
        video = open_video(str(video_path), backend="pyav")
        scene_manager = SceneManager()
        scene_manager.add_detector(det)
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

    if not scene_list:
        # Fall back to the first frame when no scene is detected.
        from scenedetect import FrameTimecode
        start = FrameTimecode(0, video.frame_rate)
        scene_list = [(start, start)]

    # Find dense-OCR boundaries within long scenes and subdivide them.
    subscene_origin: list[int] = list(range(len(scene_list)))  # tracks pre-split index per resulting segment
    n_subscenes_added = 0
    if use_dense_ocr_for_long_shots:
        long_scenes = []
        for idx, (start, end) in enumerate(scene_list):
            if end.get_seconds() - start.get_seconds() >= long_shot_threshold_sec:
                long_scenes.append({
                    "index": idx,
                    "start_sec": float(start.get_seconds()),
                    "end_sec": float(end.get_seconds()),
                })
        if long_scenes:
            if ocr_boundaries_sec is not None:
                boundaries_by_scene = _boundaries_by_scene_from_global(
                    ocr_boundaries_sec, long_scenes,
                )
            else:
                boundaries_by_scene = _call_dense_ocr_subscene_helper(
                    Path(video_path), long_scenes,
                    fps=ocr_fps,
                    threshold=ocr_subscene_threshold,
                    merge_gap=ocr_merge_gap,
                )
            # Re-derive scene_list and origin map after splitting
            new_scene_list: list = []
            new_origin: list[int] = []
            for idx, (start_tc, end_tc) in enumerate(scene_list):
                bounds = sorted(
                    b for b in boundaries_by_scene.get(idx, [])
                    if start_tc.get_seconds() < b < end_tc.get_seconds()
                )
                if not bounds:
                    new_scene_list.append((start_tc, end_tc))
                    new_origin.append(idx)
                    continue
                from scenedetect import FrameTimecode
                cur = start_tc
                for b in bounds:
                    b_tc = FrameTimecode(timecode=b, fps=video.frame_rate)
                    new_scene_list.append((cur, b_tc))
                    new_origin.append(idx)
                    cur = b_tc
                    n_subscenes_added += 1
                new_scene_list.append((cur, end_tc))
                new_origin.append(idx)
            scene_list = new_scene_list
            subscene_origin = new_origin

    # Offset each scene start while keeping the sample inside the scene.
    timestamps = []
    for start, end in scene_list:
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        ts = start_sec + offset_sec
        # Use the midpoint if the offset would exceed the scene end.
        if ts >= end_sec and end_sec > start_sec:
            ts = (start_sec + end_sec) / 2
        timestamps.append(ts)

    # Extract frames at the selected timestamps with FFmpeg.
    frames = []
    for i, ts in enumerate(timestamps):
        out_path = output_dir / f"scene_{i:04d}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-ss", str(ts),
                "-i", str(video_path),
                "-vframes", "1",
                "-vf", f"scale='if(gt(iw,ih),{max_dimension},-2)':'if(gt(ih,iw),{max_dimension},-2)'",
                "-q:v", "2",
                str(out_path),
                "-y",
            ],
            check=True,
            capture_output=True,
        )
        faces = detect_faces(out_path)
        scene_start, scene_end = scene_list[i]
        info = {
            "frame_index": i,
            "path": str(out_path),
            "timestamp_sec": round(ts, 2),
            "scene_start_sec": round(scene_start.get_seconds(), 2),
            "scene_end_sec": round(scene_end.get_seconds(), 2),
            "faces": len(faces),
            "face_details": faces,
        }
        if use_dense_ocr_for_long_shots:
            origin_idx = subscene_origin[i]
            info["origin_scene_index"] = origin_idx
            info["is_subscene"] = subscene_origin.count(origin_idx) > 1
        frames.append(info)

    return frames
