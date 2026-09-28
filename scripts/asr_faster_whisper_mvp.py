#!/usr/bin/env python3
# ASR runs as a separate process in its own Python environment.
# FFmpeg creates 16 kHz mono WAV audio, and faster-whisper segments are
# converted to program-relative JSON so the parent process stays ASR-agnostic.

"""faster-whisper ASR helper for the broadcast-news metadata pipeline.

Run this script with `.venv-asr/bin/python`. The parent process supplies the
runtime configuration for CUDA-enabled CTranslate2."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("mvp_asr_faster_whisper")


@dataclass(frozen=True)
class AsrSegment:
    start: float
    end: float
    text: str


def status(message: str) -> None:
    print(message, flush=True)


def format_seconds(value: float) -> str:
    minutes, seconds = divmod(int(round(value)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    LOG.debug("running: %s", " ".join(args))
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout[-4000:], file=sys.stderr)
        if proc.stderr:
            print(proc.stderr[-4000:], file=sys.stderr)
        proc.check_returncode()
    return proc


def nvidia_roots() -> list[Path]:
    roots: list[Path] = []
    for venv_name in (".venv-asr", ".venv"):
        lib_dir = PROJECT_ROOT / venv_name / "lib"
        roots.extend(sorted(lib_dir.glob("python*/site-packages/nvidia")))
    return roots


def preload_cuda_libs() -> None:
    candidates: list[Path] = []
    for root in nvidia_roots():
        for cuda_dir in sorted(root.glob("cu*/lib")):
            candidates.extend([
                cuda_dir / "libcublas.so.13",
                cuda_dir / "libcublasLt.so.13",
            ])
        cudnn_dir = root / "cudnn" / "lib"
        candidates.extend([
            cudnn_dir / "libcudnn.so.9",
            cudnn_dir / "libcudnn_ops.so.9",
            cudnn_dir / "libcudnn_cnn.so.9",
        ])
    candidates.append(PROJECT_ROOT / "build" / "ctranslate2-cuda" / "lib" / "libctranslate2.so.4")

    seen: set[Path] = set()
    for path in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            LOG.debug("could not preload %s: %s", path, exc)


def describe_runtime(device: str) -> None:
    status(f"  ASR python: {sys.executable}")
    status(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(empty)')}")
    import ctranslate2

    status(f"  ctranslate2: {ctranslate2.__version__} ({ctranslate2.__file__})")
    if device == "cuda":
        compute_types = sorted(ctranslate2.get_supported_compute_types("cuda"))
        status(f"  ctranslate2 CUDA compute types: {', '.join(compute_types)}")


def extract_audio(input_path: Path, wav_path: Path) -> Path:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    status(f"  extracting audio: {wav_path}")
    run_cmd([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
        "-y",
    ])
    return wav_path


def transcribe_audio(args: argparse.Namespace, wav_path: Path) -> list[AsrSegment]:
    from faster_whisper import WhisperModel

    status(f"  loading faster-whisper: model={args.model} device={args.device} compute_type={args.compute_type}")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    status("  transcribing audio; segment progress will be printed below")
    segments_iter, info = model.transcribe(
        str(wav_path),
        beam_size=args.beam_size,
        language=args.language,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": args.min_silence_duration_ms},
        no_repeat_ngram_size=3,
    )
    if getattr(info, "duration", None):
        status(f"  audio duration seen by ASR: {format_seconds(float(info.duration))}")

    segments: list[AsrSegment] = []
    last_text = ""
    for segment in segments_iter:
        text = segment.text.strip()
        if not text or text == last_text:
            continue
        segments.append(AsrSegment(start=round(segment.start, 2), end=round(segment.end, 2), text=text))
        last_text = text
        if len(segments) == 1 or len(segments) % 10 == 0:
            status(f"  [asr {len(segments)} segments] latest_end={format_seconds(segment.end)}")
    status(f"  ASR complete: {len(segments)} segments")
    return segments


def write_asr_json(output_path: Path, input_path: Path, wav_path: Path, args: argparse.Namespace, segments: list[AsrSegment]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "mvp_asr_segments_v1",
        "source_file": str(input_path),
        "audio_file": str(wav_path),
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "language": args.language,
        "segments": [asdict(segment) for segment in segments],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    status(f"  wrote ASR JSON: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run faster-whisper ASR for the metadata pipeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--min-silence-duration-ms", type=int, default=500)
    parser.add_argument("--keep-wav", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    work_dir = (args.work_dir.resolve() if args.work_dir else output_path.parent)
    work_dir.mkdir(parents=True, exist_ok=True)

    status("ASR helper")
    status(f"  input:  {input_path}")
    status(f"  output: {output_path}")
    status(f"  work:   {work_dir}")

    preload_cuda_libs()
    describe_runtime(args.device)

    wav_path = input_path if input_path.suffix.lower() == ".wav" else work_dir / f"{input_path.stem}.wav"
    if input_path.suffix.lower() != ".wav" and not wav_path.exists():
        extract_audio(input_path, wav_path)
    elif input_path.suffix.lower() != ".wav":
        status(f"  reusing audio: {wav_path}")

    segments = transcribe_audio(args, wav_path)
    write_asr_json(output_path, input_path, wav_path, args, segments)

    if not args.keep_wav and input_path.suffix.lower() != ".wav" and wav_path.exists():
        wav_path.unlink()
        status(f"  removed temporary audio: {wav_path}")


if __name__ == "__main__":
    main()
