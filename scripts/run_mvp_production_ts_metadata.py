#!/usr/bin/env python3
# ``run_pipeline`` defines the processing order, and every stage writes its
# intermediate artifacts to disk. Captions, ASR, scenes, and visual annotations
# are produced independently, then aggregated into time windows for the
# finalizer. JSON artifacts connect stages and support validated resumption.

"""Production Japanese TV news TS metadata pipeline.

The pipeline combines ARIB B24 captions, ASR, representative frames, visual
annotations, OCR/telop observations, and compact evidence windows. The final
        output conforms to `schemas/news_schema.py`."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ASR_PYTHON = PROJECT_ROOT / ".venv-asr" / "bin" / "python"
ASR_HELPER = PROJECT_ROOT / "scripts" / "asr_faster_whisper_mvp.py"
FINALIZER_PROMPT_PATH = PROJECT_ROOT / "prompts" / "program_metadata_production_openai_v4_1.txt"
FRAME_PROMPT_PATH = PROJECT_ROOT / "prompts" / "frame_analysis_openai_v2.txt"

from schemas.news_schema import (
    Domain,
    FrameVisualAnnotation,
    NonEditorialSegment,
    ProgramMetadata,
    Topic,
)

LOG = logging.getLogger("mvp_production_pipeline")
DOMAIN_VALUES = set(get_args(Domain))

# Explicit transition expressions commonly found in broadcast-news captions.
TEXT_CUE_PATTERNS = [
    r"続いて(は|の)",
    r"では次",
    r"次は",
    r"はこちらです",
    r"こんばんは。",
    r"さんです。",
    r"気象情報",
    r"マーケットを確認",
    r"特集",
    r"(ＷＢＳ|WBS)\s*(Ｑｕｉｃｋ|Quick)",
]
TEXT_CUE_RE = re.compile("|".join(TEXT_CUE_PATTERNS))

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "weather": ["天気", "気象", "気温", "降水", "予報", "台風", "梅雨", "猛暑", "週間天気"],
    "sport": ["スポーツ", "試合", "選手", "野球", "サッカー", "大谷", "得点", "優勝", "リーグ"],
    "arts_culture_entertainment_media": ["芸能", "映画", "音楽", "俳優", "タレント", "舞台", "アニメ", "ライブ"],
    "economy_business_finance": ["経済", "企業", "市場", "株価", "為替", "日経平均", "決算", "金利", "物価", "日銀", "商品", "サービス", "店舗", "投資", "半導体", "ナフサ", "円安", "決算"],
    "politics": ["首相", "政府", "国会", "大臣", "選挙", "政権", "与党", "野党", "法案", "政策"],
    "crime_law_justice": ["逮捕", "容疑者", "警察", "事件", "裁判", "判決", "起訴", "捜査", "詐欺"],
    "disaster_accident_emergency": ["地震", "津波", "火災", "事故", "避難", "被害", "救助", "大雨", "洪水", "噴火"],
    "health": ["医療", "病院", "感染", "患者", "ワクチン", "健康", "薬", "医師"],
    "science_technology": ["科学", "技術", "AI", "人工知能", "宇宙", "ロボット", "研究", "実証実験"],
    "environment": ["環境", "脱炭素", "温暖化", "再生可能", "汚染", "気候変動"],
    "education": ["学校", "教育", "大学", "受験", "授業", "生徒", "学生"],
    "conflict_war_peace": ["戦争", "停戦", "攻撃", "軍", "ミサイル", "ウクライナ", "ガザ", "紛争"],
    "labour": ["労働", "賃上げ", "雇用", "就職", "労組", "ストライキ", "人手不足"],
    "lifestyle_leisure": ["旅行", "観光", "グルメ", "暮らし", "レジャー", "家計", "消費者"],
    "society": ["社会", "地域", "交通", "鉄道", "自治体", "住民", "制度", "問題"],
}

NON_EDITORIAL_KIND_SUMMARY = {
    "commercial": "CM/ad interval",
    "program_promo": "program promotion interval",
    "sponsor": "sponsor interval",
    "opening_or_ending": "opening/ending interval",
    "other_non_editorial": "other non-editorial interval",
    "ambiguous": "ambiguous non-editorial interval",
}

SourceName = Literal["caption", "asr", "primary"]


@dataclass(frozen=True)
class TextSegment:
    source: SourceName
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    frame_index: int
    timestamp_sec: float
    path: Path


@dataclass(frozen=True)
class FrameAnnotationRecord:
    frame: FrameRecord
    span_start_sec: float
    span_end_sec: float
    annotation: FrameVisualAnnotation


@dataclass(frozen=True)
class EvidenceWindow:
    window_id: str
    start_sec: float
    end_sec: float
    captions: list[str]
    asr: list[str]
    primary_text: list[str]
    frame_ids: list[str]
    ocr_texts: list[str]
    primary_onscreen_texts: list[str]
    visual_summaries: list[str]
    vlm_presentation_hints: list[str]
    vlm_domain_hints: list[str]
    non_editorial_hints: list[str]


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    start_sec: float
    end_sec: float
    start_boundary_id: str
    end_boundary_id: str
    representative_frame_ids: list[str]


@dataclass(frozen=True)
class CueRecord:
    cue_id: str
    start_sec: float
    end_sec: float
    text: str
    cue_type: str
    strength: str
    confidence: float


@dataclass(frozen=True)
class BoundaryCandidateRecord:
    boundary_id: str
    time_sec: float
    boundary_type: str
    source: str
    score: float
    reasons: list[str]


# Small helpers shared by stages for progress, timing, and subprocesses.

def status(message: str) -> None:
    print(message, flush=True)


def stage(current: int, total: int, title: str) -> None:
    status(f"\n[{current}/{total}] {title}")


def progress(label: str, current: int, total: int, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    status(f"  [{label} {current}/{total}]{suffix}")


def format_seconds(value: float) -> str:
    minutes, seconds = divmod(int(round(value)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    LOG.debug("running: %s", " ".join(args))
    proc = subprocess.run(args, check=True, capture_output=True, text=True)
    return proc


def ffprobe_duration(input_path: Path) -> float:
    proc = run_cmd([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ])
    return float(proc.stdout.strip())


def logical_program_duration(args: argparse.Namespace, source_duration_sec: float) -> float:
    """Return a zero-based logical program duration without changing source media."""
    start = args.logical_start_sec
    end = args.logical_end_sec
    if start is None and end is None:
        return source_duration_sec
    if start is None or end is None:
        raise ValueError("--logical-start-sec and --logical-end-sec must be specified together")
    if start < 0 or end <= start:
        raise ValueError("logical program window must satisfy 0 <= start < end")
    if end > source_duration_sec + 1.0:
        raise ValueError(
            f"logical program end {end:.3f}s exceeds source duration {source_duration_sec:.3f}s"
        )
    return end - start


def dedupe(values: list[str], limit: int | None = None) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def truncate_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "..."


def span_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def segment_in_span(segment: TextSegment, start: float, end: float) -> bool:
    return span_overlap(segment.start, segment.end, start, end) > 0


def cuda_library_dirs() -> list[Path]:
    dirs = [PROJECT_ROOT / "build" / "ctranslate2-cuda" / "lib"]
    for venv_name in (".venv-asr", ".venv"):
        for nvidia_root in (PROJECT_ROOT / venv_name / "lib").glob("python*/site-packages/nvidia"):
            dirs.extend(sorted(nvidia_root.glob("cu*/lib")))
            dirs.append(nvidia_root / "cudnn" / "lib")

    seen: set[Path] = set()
    existing: list[Path] = []
    for path in dirs:
        resolved = path.resolve()
        if path.exists() and resolved not in seen:
            existing.append(path)
            seen.add(resolved)
    return existing


def asr_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    entries = [str(path) for path in cuda_library_dirs()]
    entries.extend(item for item in env.get("LD_LIBRARY_PATH", "").split(":") if item)
    merged: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry not in seen:
            merged.append(entry)
            seen.add(entry)
    env["LD_LIBRARY_PATH"] = ":".join(merged)
    env["PYTHONUNBUFFERED"] = "1"
    return env


# Normalize captions and ASR into TextSegment records and build the primary timeline.

def load_caption_segments(path: Path, *, caption_lag_sec: float) -> list[TextSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = data.get("segments", data if isinstance(data, list) else [])
    segments: list[TextSegment] = []
    for item in raw_segments:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        raw_start = float(item.get("start_sec", item.get("start", 0.0)))
        raw_end = float(item.get("end_sec", item.get("end", raw_start)))
        start = max(0.0, raw_start - caption_lag_sec)
        end = max(start + 0.1, raw_end - caption_lag_sec)
        segments.append(TextSegment("caption", round(start, 2), round(end, 2), text, item.get("speaker")))
    status(f"  loaded captions: {len(segments)} segments from {path}")
    return segments


def extract_or_load_captions(args: argparse.Namespace, input_path: Path, work_dir: Path) -> tuple[list[TextSegment], Path | None, list[str]]:
    warnings: list[str] = []
    captions_dir = work_dir / "captions"
    canonical_path = captions_dir / f"{input_path.stem}.captions.json"
    if args.captions_json:
        json_path = copy_json_for_resume(args.captions_json, canonical_path, "captions JSON")
        return load_caption_segments(json_path, caption_lag_sec=args.caption_lag_sec), json_path, warnings
    if args.caption_mode == "skip":
        status("  captions skipped by explicit CLI option")
        warnings.append("Caption extraction was explicitly skipped.")
        return [], None, warnings
    if args.resume:
        resume_path = canonical_path if canonical_path.exists() else summary_artifact_path(work_dir.parent, "captions_json")
        if resume_path:
            json_path = copy_json_for_resume(resume_path, canonical_path, "captions JSON")
            status(f"  resume: reusing captions from {json_path}")
            return load_caption_segments(json_path, caption_lag_sec=args.caption_lag_sec), json_path, warnings

    try:
        from src.extract_captions import extract_captions

        status("  extracting ARIB B24 captions with assdumper")
        status(f"  caption lag correction: -{args.caption_lag_sec:.1f}s")
        json_path = extract_captions(
            ts_path=input_path,
            out_dir=captions_dir,
            sid=args.caption_sid,
            accurate=args.caption_accurate,
            keep_ass=args.keep_ass,
        )
        segments = load_caption_segments(json_path, caption_lag_sec=args.caption_lag_sec)
        return segments, json_path, warnings
    except Exception as exc:
        message = f"caption extraction failed: {exc}"
        if not args.allow_partial:
            raise RuntimeError(message) from exc
        warnings.append("HIGH PRIORITY: " + message)
        status(f"  warning: {message}")
        status("  continuing because --allow-partial is enabled")
        return [], None, warnings


def load_asr_segments(path: Path) -> list[TextSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = data.get("segments", data if isinstance(data, list) else [])
    segments: list[TextSegment] = []
    for item in raw_segments:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(TextSegment(
            "asr",
            round(float(item.get("start", item.get("start_sec", 0.0))), 2),
            round(float(item.get("end", item.get("end_sec", item.get("start", 0.0)))), 2),
            text,
            None,
        ))
    status(f"  loaded ASR: {len(segments)} segments from {path}")
    return segments


def run_faster_whisper_asr(args: argparse.Namespace, input_path: Path, work_dir: Path) -> Path:
    if not ASR_PYTHON.exists():
        raise FileNotFoundError(f"ASR Python not found: {ASR_PYTHON}")
    if not ASR_HELPER.exists():
        raise FileNotFoundError(f"ASR helper not found: {ASR_HELPER}")

    asr_path = work_dir / f"{input_path.stem}.asr.json"
    cmd = [
        str(ASR_PYTHON),
        str(ASR_HELPER),
        "--input",
        str(input_path),
        "--output",
        str(asr_path),
        "--work-dir",
        str(work_dir),
        "--model",
        args.asr_model,
        "--device",
        args.asr_device,
        "--compute-type",
        args.asr_compute_type,
    ]
    if args.keep_wav:
        cmd.append("--keep-wav")
    status(f"  launching ASR helper: {ASR_PYTHON}")
    status(f"  ASR output: {asr_path}")
    for lib_dir in cuda_library_dirs():
        status(f"    CUDA lib: {lib_dir}")
    subprocess.run(cmd, check=True, env=asr_subprocess_env())
    return asr_path


def extract_or_load_asr(args: argparse.Namespace, input_path: Path, work_dir: Path) -> tuple[list[TextSegment], Path | None]:
    canonical_path = work_dir / f"{input_path.stem}.asr.json"
    if args.asr_json:
        asr_path = copy_json_for_resume(args.asr_json, canonical_path, "ASR JSON")
        return load_asr_segments(asr_path), asr_path
    if args.asr_mode == "skip":
        status("  ASR skipped")
        return [], None
    if args.resume:
        resume_path = canonical_path if canonical_path.exists() else summary_artifact_path(work_dir.parent, "asr_json")
        if resume_path:
            asr_path = copy_json_for_resume(resume_path, canonical_path, "ASR JSON")
            status(f"  resume: reusing ASR from {asr_path}")
            return load_asr_segments(asr_path), asr_path
    asr_path = run_faster_whisper_asr(args, input_path, work_dir)
    return load_asr_segments(asr_path), asr_path


def build_primary_segments(
    asr_segments: list[TextSegment],
    caption_segments: list[TextSegment],
    fused_segments: list[TextSegment] | None = None,
) -> tuple[list[TextSegment], list[str]]:
    warnings: list[str] = []
    if fused_segments:
        status("  primary text timeline: ch140 fusion (ASR timing + caption lexicon)")
        return list(fused_segments), warnings
    if caption_segments:
        if asr_segments:
            status("  primary text timeline: captions; ASR retained only as fallback/audio evidence")
        else:
            status("  primary text timeline: captions only; times use configured live-caption lag correction")
            warnings.append("ASR was unavailable; caption timestamps were used after configured lag correction.")
        return [TextSegment("primary", s.start, s.end, s.text, s.speaker) for s in caption_segments], warnings
    if asr_segments:
        status("  primary text timeline: ASR only because captions are unavailable")
        warnings.append("Captions were unavailable; ASR text was used as the primary text timeline.")
        return [TextSegment("primary", s.start, s.end, s.text, None) for s in asr_segments], warnings
    warnings.append("No ASR or captions were available; final boundaries can only use visual intervals.")
    return [], warnings


def run_ch140_fusion(
    args: argparse.Namespace,
    asr_segments: list[TextSegment],
    caption_segments: list[TextSegment],
    out_dir: Path,
    program_id: str,
    duration_sec: float,
) -> tuple[list[TextSegment], Path | None, list[str]]:
    """Fuse ASR with captions, falling back to caption priority on failure."""
    warnings: list[str] = []
    if not caption_segments:
        message = "ch140_fusion skipped: captions unavailable (no lexical dictionary); using default text merge"
        status(f"  {message}")
        warnings.append(message)
        return [], None, warnings
    if not asr_segments:
        message = "ch140_fusion skipped: ASR unavailable; captions remain the primary timeline"
        status(f"  {message}")
        warnings.append(message)
        return [], None, warnings

    fusion_dir = out_dir / "transcript_unified"
    fusion_path = fusion_dir / f"{program_id}.transcript_unified.json"
    backup_dir = fusion_dir / "backup"

    if args.resume and fusion_path.exists():
        try:
            data = json.loads(fusion_path.read_text(encoding="utf-8"))
            fused = [
                TextSegment("primary", float(s["start"]), float(s["end"]), str(s["text"]), None)
                for s in data.get("segments", [])
                if str(s.get("text", "")).strip()
            ]
            if fused:
                status(f"  resume: reusing fused transcript from {fusion_path} ({len(fused)} segments)")
                return fused, fusion_path, warnings
        except Exception as exc:
            status(f"  resume: failed to reuse fused transcript ({exc}); re-running fusion")

    import src.transcript_unified as transcript_unified

    asr_dicts = [
        {"id": i, "start": s.start, "end": s.end, "text": s.text}
        for i, s in enumerate(asr_segments)
    ]
    caption_entries = [
        transcript_unified.CaptionEntry(start=s.start, end=s.end, text=s.text)
        for s in caption_segments
    ]
    status(f"  ch140 fusion: {len(asr_dicts)} ASR segments x {len(caption_entries)} caption entries")
    status(f"  fusion model: {args.fusion_model} (window ±{args.fusion_caption_window_sec:.0f}s, glossary={'on' if args.fusion_glossary else 'off'})")
    try:
        result = transcript_unified.refine_transcript_segments(
            asr_dicts,
            caption_entries,
            out_path=fusion_path,
            program_id=program_id,
            duration_sec=duration_sec,
            backup_dir=backup_dir,
            model=args.fusion_model,
            batch_target_tokens=args.fusion_batch_target_tokens,
            max_completion_tokens=args.fusion_max_completion_tokens,
            caption_window_sec=args.fusion_caption_window_sec,
            use_glossary=args.fusion_glossary,
            glossary_path=fusion_dir / "glossary.json",
        )
    except Exception as exc:
        message = f"ch140_fusion failed; falling back to caption-priority text merge: {exc}"
        status(f"  warning: {message}")
        warnings.append(message)
        return [], None, warnings

    fused = [
        TextSegment("primary", s.start, s.end, s.text, None)
        for s in result.segments
        if s.text.strip()
    ]
    status(
        f"  fusion done: {result.n_batches} batches"
        f" ({result.n_fallback_batches} fallback), {len(fused)} segments"
    )
    if result.n_fallback_batches:
        warnings.append(
            f"ch140_fusion: {result.n_fallback_batches}/{result.n_batches} batches fell back to raw ASR text."
        )
    return fused, fusion_path, warnings


def run_topic_boundary_hints(
    args: argparse.Namespace,
    primary_segments: list[TextSegment],
    boundary_candidates: list[BoundaryCandidateRecord],
    non_editorial: list[tuple[float, float, str, list[str]]],
    llm_dir: Path,
) -> tuple[list[dict], Path | None, list[str]]:
    """Detect topic-boundary hints, returning no hints on failure."""
    warnings: list[str] = []
    if not primary_segments:
        message = "boundary hints skipped: no primary text segments"
        status(f"  {message}")
        warnings.append(message)
        return [], None, warnings

    out_path = llm_dir / "topic_boundaries.json"
    if args.resume and out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            hints = data.get("boundaries", [])
            status(f"  resume: reusing topic boundary hints from {out_path} ({len(hints)} hints)")
            return hints, out_path, warnings
        except Exception as exc:
            status(f"  resume: failed to reuse boundary hints ({exc}); re-running")

    import src.topic_boundary_llm as topic_boundary_llm

    segments = [{"start": s.start, "end": s.end, "text": s.text} for s in primary_segments]
    # Snap only to visual scene/OCR candidates. Text cues are not independent
    # visual evidence, and commercial intervals are handled by filtering.
    candidate_pairs = [
        (c.time_sec, c.boundary_id) for c in boundary_candidates
        if c.source not in ("cue", "cm_detector")
    ]
    ne_intervals = [(start, end) for start, end, _kind, _frames in non_editorial]
    status(f"  boundary hints: {len(segments)} segments, {len(candidate_pairs)} visual candidates")
    status(f"  boundary model: {args.boundary_model}")
    try:
        hints = topic_boundary_llm.detect_topic_boundaries(
            segments,
            out_path=out_path,
            backup_dir=llm_dir / "topic_boundaries_backup",
            model=args.boundary_model,
            chunk_target_tokens=args.boundary_chunk_target_tokens,
            chunk_overlap_sec=args.boundary_chunk_overlap_sec,
            merge_tolerance_sec=args.boundary_merge_tolerance_sec,
            snap_tolerance_sec=args.boundary_snap_tolerance_sec,
            min_gap_sec=args.min_topic_duration_sec,
            candidates=candidate_pairs,
            non_editorial=ne_intervals,
        )
    except Exception as exc:
        message = f"topic boundary hints failed; continuing without hints: {exc}"
        status(f"  warning: {message}")
        warnings.append(message)
        return [], None, warnings

    status(f"  boundary hints: {len(hints)} final boundaries")
    return [asdict(h) for h in hints], out_path, warnings


# Save representative frames and obtain or restore visual annotations.

def extract_representative_frames(
    input_path: Path,
    frames_dir: Path,
    *,
    duration_sec: float,
    interval_sec: float,
    max_frames: int | None,
    max_dimension: int,
) -> list[FrameRecord]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    timestamps: list[float] = []
    current = min(5.0, max(duration_sec / 2.0, 0.0))
    while current < duration_sec:
        timestamps.append(round(current, 2))
        current += interval_sec
        if max_frames is not None and len(timestamps) >= max_frames:
            break
    if not timestamps and duration_sec >= 0:
        timestamps = [0.0]

    status(f"  extracting {len(timestamps)} representative frames to {frames_dir}")
    frames: list[FrameRecord] = []
    for idx, ts in enumerate(timestamps):
        frame_id = f"frame_{idx:04d}"
        out_path = frames_dir / f"{frame_id}.jpg"
        progress("frames", idx + 1, len(timestamps), f"{format_seconds(ts)} -> {out_path.name}")
        run_cmd([
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(ts),
            "-i",
            str(input_path),
            "-vframes",
            "1",
            "-vf",
            f"scale='if(gt(iw,ih),{max_dimension},-2)':'if(gt(ih,iw),{max_dimension},-2)'",
            "-q:v",
            "2",
            str(out_path),
            "-y",
        ])
        frames.append(FrameRecord(frame_id=frame_id, frame_index=idx, timestamp_sec=ts, path=out_path))
    return frames


def encode_image_base64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def openai_client():
    from openai import OpenAI

    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    kwargs: dict[str, Any] = {"api_key": api_key}
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if endpoint:
        kwargs["base_url"] = endpoint
    return OpenAI(**kwargs)


def scene_id_for_frame(frame: FrameRecord) -> str | None:
    if frame.frame_id.startswith("scene_") and "_" in frame.frame_id:
        return frame.frame_id.rsplit("_", 1)[0]
    return None


def annotate_frame_openai(frame: FrameRecord, *, model: str, prompt: str) -> FrameVisualAnnotation:
    metadata = {
        "frame_id": frame.frame_id,
        "scene_id": scene_id_for_frame(frame),
        "timestamp_sec": frame.timestamp_sec,
        "frame_index": frame.frame_index,
    }
    metadata_prompt = (
        prompt
        + "\n\n## フレーム識別情報 (必須)\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\nこの識別情報の値を出力 JSON にそのまま含めてください。"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": metadata_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_base64(frame.path)}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]
    # Text-dense frames may reach the structured-output limit. Retry once with
    # a larger budget only after truncation; fail with context if the retry is
    # also truncated instead of fabricating an annotation.
    from openai import LengthFinishReasonError

    client = openai_client()
    last_error: LengthFinishReasonError | None = None
    for max_completion_tokens in (4096, 16384):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                max_completion_tokens=max_completion_tokens,
                messages=messages,
                response_format=FrameVisualAnnotation,
            )
            break
        except LengthFinishReasonError as exc:
            last_error = exc
            status(
                f"  VLM output hit the {max_completion_tokens}-token limit for {frame.frame_id}"
                + ("; retrying with a higher limit" if max_completion_tokens < 16384 else "")
            )
    else:
        raise RuntimeError(
            f"VLM annotation for {frame.frame_id} exceeded the completion-token limit even at 16384"
        ) from last_error
    message = response.choices[0].message
    if message.refusal:
        raise RuntimeError(f"VLM refused frame {frame.frame_id}: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError(f"VLM returned no parsed annotation for {frame.frame_id}")
    payload = message.parsed.model_dump(mode="json")
    payload.update(metadata)
    return FrameVisualAnnotation.model_validate(payload)


def frame_time_spans(frames: list[FrameRecord], duration_sec: float) -> list[tuple[float, float]]:
    if not frames:
        return []
    times = [frame.timestamp_sec for frame in frames]
    spans: list[tuple[float, float]] = []
    for idx, ts in enumerate(times):
        start = 0.0 if idx == 0 else (times[idx - 1] + ts) / 2.0
        end = duration_sec if idx == len(times) - 1 else (ts + times[idx + 1]) / 2.0
        spans.append((round(start, 2), round(end, 2)))
    return spans


def save_annotations(records: list[FrameAnnotationRecord], annotations_path: Path) -> None:
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "frame_id": record.frame.frame_id,
                    "image_path": str(record.frame.path),
                    "span_start_sec": record.span_start_sec,
                    "span_end_sec": record.span_end_sec,
                    "annotation": record.annotation.model_dump(mode="json"),
                }
                for record in records
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_annotations(path: Path) -> list[FrameAnnotationRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records: list[FrameAnnotationRecord] = []
    for idx, item in enumerate(raw):
        annotation = FrameVisualAnnotation.model_validate(item.get("annotation", item))
        frame = FrameRecord(
            frame_id=str(item.get("frame_id") or annotation.frame_id),
            frame_index=int(annotation.frame_index if annotation.frame_index is not None else idx),
            timestamp_sec=float(annotation.timestamp_sec if annotation.timestamp_sec is not None else 0.0),
            path=Path(item.get("image_path") or ""),
        )
        records.append(FrameAnnotationRecord(
            frame=frame,
            span_start_sec=float(item.get("span_start_sec", annotation.timestamp_sec or 0.0)),
            span_end_sec=float(item.get("span_end_sec", annotation.timestamp_sec or 0.0)),
            annotation=annotation,
        ))
    status(f"  loaded frame annotations: {len(records)} from {path}")
    return records


def annotate_or_load_frames(
    args: argparse.Namespace,
    frames: list[FrameRecord],
    duration_sec: float,
    annotations_path: Path,
) -> list[FrameAnnotationRecord]:
    if args.frame_annotations and args.frame_annotations.exists():
        records = load_annotations(args.frame_annotations)
        if args.frame_annotations.resolve() != annotations_path.resolve():
            save_annotations(records, annotations_path)
            status(f"  copied frame annotations into run directory: {annotations_path}")
        return records
    if (args.reuse_annotations or args.resume) and annotations_path.exists():
        status(f"  resume: reusing frame annotations from {annotations_path}")
        return load_annotations(annotations_path)
    if args.vlm_mode == "skip":
        status("  VLM frame annotation skipped")
        return []

    missing = [str(frame.path) for frame in frames if not frame.path.exists() or frame.path.stat().st_size == 0]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" ... and {len(missing) - 5} more"
        raise FileNotFoundError(f"representative frame file(s) missing before VLM annotation: {preview}{suffix}")

    prompt = FRAME_PROMPT_PATH.read_text(encoding="utf-8")
    spans = frame_time_spans(frames, duration_sec)
    concurrency = max(1, args.vlm_concurrency)
    status(f"  annotating {len(frames)} frames with {args.vlm_model}"
           + (f" (concurrency={concurrency})" if concurrency > 1 else ""))

    def annotate_one(idx: int) -> tuple[int, FrameAnnotationRecord]:
        frame = frames[idx]
        start, end = spans[idx]
        annotation = annotate_frame_openai(frame, model=args.vlm_model, prompt=prompt)
        return idx, FrameAnnotationRecord(
            frame=frame, span_start_sec=start, span_end_sec=end, annotation=annotation
        )

    def report(idx: int, record: FrameAnnotationRecord, done: int) -> None:
        frame, annotation = record.frame, record.annotation
        progress("vlm", done, len(frames), f"{frame.frame_id} at {format_seconds(frame.timestamp_sec)}")
        status(
            "    -> "
            f"shot={annotation.shot_type} "
            f"presentation_hint={annotation.likely_presentation_form} "
            f"non_editorial={annotation.non_editorial_kind}"
        )

    if concurrency == 1:
        records = []
        for idx in range(len(frames)):
            _, record = annotate_one(idx)
            report(idx, record, len(records) + 1)
            records.append(record)
    else:
        # Frame annotations are independent. Collect them as they complete,
        # then restore source order so output matches sequential execution.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        by_index: dict[int, FrameAnnotationRecord] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(annotate_one, idx) for idx in range(len(frames))]
            for future in as_completed(futures):
                idx, record = future.result()
                by_index[idx] = record
                report(idx, record, len(by_index))
        records = [by_index[idx] for idx in range(len(frames))]
    save_annotations(records, annotations_path)
    status(f"  wrote frame annotations: {annotations_path}")
    return records


# Derive non-editorial intervals and deterministic fallback segmentation.

def is_non_editorial(annotation: FrameVisualAnnotation) -> bool:
    if annotation.non_editorial_kind != "none" and not annotation.is_probably_editorial:
        return True
    if annotation.non_editorial_kind in {"commercial", "program_promo", "opening_or_ending", "other_non_editorial"}:
        return True
    if annotation.sponsor_presentation == "fullscreen_sponsor_credit":
        return True
    return False


def merge_non_editorial(intervals: list[tuple[float, float, str, list[str]]], *, max_gap_sec: float = 5.0) -> list[tuple[float, float, str, list[str]]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda item: (item[0], item[1], item[2]))
    merged: list[tuple[float, float, str, list[str]]] = []
    for start, end, kind, frame_ids in intervals:
        if not merged:
            merged.append((start, end, kind, list(frame_ids)))
            continue
        prev_start, prev_end, prev_kind, prev_frames = merged[-1]
        if kind == prev_kind and start <= prev_end + max_gap_sec:
            merged[-1] = (prev_start, max(prev_end, end), kind, dedupe([*prev_frames, *frame_ids]))
        else:
            merged.append((start, end, kind, list(frame_ids)))
    return merged


def non_editorial_intervals(records: list[FrameAnnotationRecord]) -> list[tuple[float, float, str, list[str]]]:
    intervals: list[tuple[float, float, str, list[str]]] = []
    for record in records:
        annotation = record.annotation
        if not is_non_editorial(annotation):
            continue
        kind = annotation.non_editorial_kind
        if kind == "none" and annotation.sponsor_presentation == "fullscreen_sponsor_credit":
            kind = "sponsor"
        if kind == "none":
            kind = "ambiguous"
        intervals.append((record.span_start_sec, record.span_end_sec, kind, [record.frame.frame_id]))
    return merge_non_editorial(intervals)


def build_non_editorial_segments(intervals: list[tuple[float, float, str, list[str]]]) -> list[NonEditorialSegment]:
    segments: list[NonEditorialSegment] = []
    for idx, (start, end, kind, frame_ids) in enumerate(intervals):
        segments.append(NonEditorialSegment(
            segment_id=f"ne{idx:03d}",
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            non_editorial_kind=kind,  # type: ignore[arg-type]
            summary=NON_EDITORIAL_KIND_SUMMARY.get(kind, "non-editorial interval"),
            source_scene_ids=dedupe(frame_ids),
            representative_frame_ids=dedupe(frame_ids),
            notes="Detected from visual-only frame evidence; kept outside Topic output.",
        ))
    return segments


def records_in_span(records: list[FrameAnnotationRecord], start: float, end: float) -> list[FrameAnnotationRecord]:
    return [record for record in records if span_overlap(record.span_start_sec, record.span_end_sec, start, end) > 0]


def text_in_span(segments: list[TextSegment], start: float, end: float) -> list[TextSegment]:
    return [segment for segment in segments if segment_in_span(segment, start, end)]


def text_cue_boundaries(primary_segments: list[TextSegment], *, min_gap_sec: float, duration_sec: float) -> list[float]:
    boundaries: list[float] = []
    last = 0.0
    for segment in primary_segments:
        if segment.start <= 0.5 or segment.start >= duration_sec - 0.5:
            continue
        if segment.start - last < min_gap_sec:
            continue
        if TEXT_CUE_RE.search(segment.text):
            boundaries.append(round(segment.start, 2))
            last = segment.start
    return boundaries


def editorial_spans(duration_sec: float, non_editorial: list[tuple[float, float, str, list[str]]], cues: list[float]) -> list[tuple[float, float]]:
    cuts = {0.0, round(duration_sec, 2)}
    for start, end, _kind, _frames in non_editorial:
        cuts.add(round(max(0.0, start), 2))
        cuts.add(round(min(duration_sec, end), 2))
    for cue in cues:
        if 0.0 < cue < duration_sec:
            cuts.add(round(cue, 2))
    sorted_cuts = sorted(cuts)
    spans: list[tuple[float, float]] = []
    for start, end in zip(sorted_cuts, sorted_cuts[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        inside_non_editorial = any(ne_start <= midpoint < ne_end for ne_start, ne_end, _kind, _frames in non_editorial)
        if not inside_non_editorial:
            spans.append((start, end))
    return spans


def frame_ids_for_records(records: list[FrameAnnotationRecord]) -> list[str]:
    return dedupe([record.frame.frame_id for record in records])


def evidence_texts_for_span(
    primary: list[TextSegment],
    captions: list[TextSegment],
    asr: list[TextSegment],
    records: list[FrameAnnotationRecord],
    *,
    limit: int = 24,
) -> list[str]:
    values: list[str] = []
    if primary:
        values.extend(f"primary: {truncate_text(s.text, 180)}" for s in primary[:8])
    elif captions:
        values.extend(f"caption: {truncate_text(s.text, 180)}" for s in captions[:8])
    else:
        values.extend(f"asr: {truncate_text(s.text, 180)}" for s in asr[:8])
    for record in records[:8]:
        annotation = record.annotation
        values.extend(f"ocr: {truncate_text(text, 120)}" for text in annotation.ocr_texts[:5])
        if annotation.primary_onscreen_text:
            values.append(f"telop: {truncate_text(annotation.primary_onscreen_text, 160)}")
        values.append(f"visual: {truncate_text(annotation.visual_summary, 180)}")
    return dedupe(values, limit=limit)


def span_text(primary: list[TextSegment], captions: list[TextSegment], asr: list[TextSegment], records: list[FrameAnnotationRecord]) -> str:
    values = evidence_texts_for_span(primary, captions, asr, records, limit=None)
    return "\n".join(values)


def classify_domain(text: str, records: list[FrameAnnotationRecord]) -> tuple[str, list[str], float, str | None]:
    scores: dict[str, int] = {}
    lower = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword.lower() in lower)
        if count:
            scores[domain] = count
    hints = dedupe([
        hint
        for record in records
        for hint in record.annotation.domain_hints
        if hint in DOMAIN_VALUES
    ])
    if scores:
        ranked = [domain for domain, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        domain = ranked[0]
        alternatives = [item for item in dedupe([*ranked[1:], *hints], limit=3) if item != domain]
        return domain, alternatives, 0.74, None
    if hints and text.strip():
        domain = hints[0]
        alternatives = [item for item in hints[1:4] if item != domain]
        return domain, alternatives, 0.50, "domain used visual hints because text evidence was weak"
    return "society", [], 0.34, "domain fell back to society because text evidence was ambiguous"


def title_for_span(primary: list[TextSegment], captions: list[TextSegment], records: list[FrameAnnotationRecord], fallback: str) -> str:
    for record in records:
        text = record.annotation.primary_onscreen_text
        if text:
            return truncate_text(text, 60)
    for segment in [*primary, *captions]:
        if segment.text.strip():
            return truncate_text(segment.text, 60)
    return fallback


def summary_for_span(primary: list[TextSegment], captions: list[TextSegment], records: list[FrameAnnotationRecord]) -> str:
    text = "".join(segment.text for segment in primary[:8]).strip()
    if not text:
        text = "".join(segment.text for segment in captions[:8]).strip()
    if text:
        return truncate_text(text, 240)
    visuals = "。".join(record.annotation.visual_summary for record in records[:3]).strip()
    return truncate_text(visuals, 240) if visuals else "Editorial segment with limited text evidence."


def build_heuristic_metadata(
    *,
    input_path: Path,
    duration_sec: float,
    program_id: str,
    channel_name: str | None,
    program_name: str | None,
    broadcast_start: str | None,
    broadcast_end: str | None,
    primary_segments: list[TextSegment],
    caption_segments: list[TextSegment],
    asr_segments: list[TextSegment],
    annotation_records: list[FrameAnnotationRecord],
    non_editorial: list[tuple[float, float, str, list[str]]],
    min_topic_duration_sec: float,
    initial_warnings: list[str],
) -> ProgramMetadata:
    cues = text_cue_boundaries(primary_segments, min_gap_sec=min_topic_duration_sec, duration_sec=duration_sec)
    if not primary_segments and annotation_records:
        initial_warnings.append("Heuristic fallback had no text timeline; it used visual frame intervals only for coarse editorial spans.")
    spans = editorial_spans(duration_sec, non_editorial, cues)
    status(f"  heuristic text cue boundaries: {len(cues)}")
    status(f"  heuristic editorial spans: {len(spans)}")

    topics: list[Topic] = []
    for idx, (start, end) in enumerate(spans):
        span_primary = text_in_span(primary_segments, start, end)
        span_captions = text_in_span(caption_segments, start, end)
        span_asr = text_in_span(asr_segments, start, end)
        span_records = records_in_span(annotation_records, start, end)
        if not span_primary and not span_captions and not span_asr and not span_records:
            continue

        text = span_text(span_primary, span_captions, span_asr, span_records)
        domain, alternatives, domain_conf, domain_note = classify_domain(text, span_records)
        topic_id = f"t{idx:03d}"
        evidence = evidence_texts_for_span(span_primary, span_captions, span_asr, span_records)
        title = title_for_span(span_primary, span_captions, span_records, f"Editorial topic {idx:03d}")
        summary = summary_for_span(span_primary, span_captions, span_records)
        topics.append(Topic(
            topic_id=topic_id,
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            editorial=True,
            domain=domain,  # type: ignore[arg-type]
            domain_alternatives=alternatives,  # type: ignore[arg-type]
            title=title,
            summary=summary,
            interrupted_by_non_editorial_segment_ids=[],
            evidence_texts=evidence,
            notes=domain_note,
        ))

    return ProgramMetadata(
        program_id=program_id,
        source_ts_path=str(input_path),
        channel_name=channel_name,
        program_name=program_name,
        broadcast_start=broadcast_start,
        broadcast_end=broadcast_end,
        topics=topics,
        non_editorial_segments=build_non_editorial_segments(non_editorial),
        warnings=dedupe(initial_warnings),
    )


# Aggregate evidence into time windows, finalize, and normalize the output.

def build_evidence_windows(
    *,
    duration_sec: float,
    window_sec: float,
    captions: list[TextSegment],
    asr: list[TextSegment],
    primary: list[TextSegment],
    annotations: list[FrameAnnotationRecord],
    non_editorial: list[tuple[float, float, str, list[str]]],
    extra_cuts: list[float] | None = None,
) -> list[EvidenceWindow]:
    cuts = {0.0, round(duration_sec, 2)}
    cursor = 0.0
    while cursor < duration_sec:
        cuts.add(round(cursor, 2))
        cursor += window_sec
    for start, end, _kind, _frames in non_editorial:
        cuts.add(round(max(0.0, start), 2))
        cuts.add(round(min(duration_sec, end), 2))
    for segment in primary:
        if TEXT_CUE_RE.search(segment.text):
            cuts.add(round(segment.start, 2))
    for cut in extra_cuts or []:
        if 0.0 < cut < duration_sec:
            cuts.add(round(cut, 2))
    sorted_cuts = sorted(cuts)

    windows: list[EvidenceWindow] = []
    for idx, (start, end) in enumerate(zip(sorted_cuts, sorted_cuts[1:])):
        if end <= start:
            continue
        win_captions = text_in_span(captions, start, end)
        win_asr = text_in_span(asr, start, end)
        win_primary = text_in_span(primary, start, end)
        win_records = records_in_span(annotations, start, end)
        ne_hints = [kind for ne_start, ne_end, kind, _frames in non_editorial if span_overlap(ne_start, ne_end, start, end) > 0]
        caption_preferred_asr = [] if captions and (win_captions or win_primary) else win_asr
        window = EvidenceWindow(
            window_id=f"w{idx:04d}",
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            captions=dedupe([truncate_text(seg.text, 180) for seg in win_captions], limit=12),
            asr=dedupe([truncate_text(seg.text, 180) for seg in caption_preferred_asr], limit=12),
            primary_text=dedupe([truncate_text(seg.text, 180) for seg in win_primary], limit=12),
            frame_ids=frame_ids_for_records(win_records),
            ocr_texts=dedupe([truncate_text(text, 120) for record in win_records for text in record.annotation.ocr_texts], limit=20),
            primary_onscreen_texts=dedupe([truncate_text(record.annotation.primary_onscreen_text or "", 160) for record in win_records], limit=8),
            visual_summaries=dedupe([truncate_text(record.annotation.visual_summary, 180) for record in win_records], limit=8),
            vlm_presentation_hints=dedupe([record.annotation.likely_presentation_form for record in win_records], limit=6),
            vlm_domain_hints=dedupe([hint for record in win_records for hint in record.annotation.domain_hints], limit=6),
            non_editorial_hints=dedupe(ne_hints, limit=6),
        )
        if any([window.captions, window.asr, window.primary_text, window.frame_ids, window.non_editorial_hints]):
            windows.append(window)
    status(f"  evidence windows: {len(windows)}")
    return windows


def write_evidence_json(
    path: Path,
    *,
    input_path: Path,
    duration_sec: float,
    captions_path: Path | None,
    asr_path: Path | None,
    annotations_path: Path | None,
    windows: list[EvidenceWindow],
    non_editorial: list[tuple[float, float, str, list[str]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mvp_production_evidence_v1",
        "source_ts_path": str(input_path),
        "duration_sec": duration_sec,
        "captions_json": str(captions_path) if captions_path else None,
        "asr_json": str(asr_path) if asr_path else None,
        "frame_annotations_json": str(annotations_path) if annotations_path else None,
        "non_editorial_intervals": [
            {"start_sec": start, "end_sec": end, "kind": kind, "representative_frame_ids": frame_ids}
            for start, end, kind, frame_ids in non_editorial
        ],
        "windows": [asdict(window) for window in windows],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    status(f"  wrote evidence timeline: {path}")


def finalizer_input_payload(
    *,
    input_path: Path,
    duration_sec: float,
    program_id: str,
    channel_name: str | None,
    program_name: str | None,
    broadcast_start: str | None,
    broadcast_end: str | None,
    windows: list[EvidenceWindow],
    non_editorial_segments: list[NonEditorialSegment],
    warnings: list[str],
    max_windows: int,
    text_merge_mode: str = "caption_priority",
    boundary_hints: list[dict] | None = None,
) -> dict[str, Any]:
    selected_windows = windows[:max_windows]
    omitted = max(0, len(windows) - len(selected_windows))
    hard_rules = [
        "最終出力は ProgramMetadata schema に適合する構造化データのみとする。",
        "VLMラベルは視覚のみの証拠として扱い、VLMラベル単独で最終Topic境界を決めない。",
        "CM・番組宣伝・純粋な提供画面・オープニング/エンディングはTopicに含めず、non_editorial_segments に保持する。",
        "non_editorial_segments の representative_frame_ids は視覚証拠として保持する。",
        "時間窓のstart/endは証拠の集約区切りであり証拠ではない。Topicのstart_sec/end_secには本文セグメント時刻・話者交代・scene境界・OCR変化のいずれかの証拠時刻だけを使い、窓の端の時刻をそのまま使わない。",
    ]
    if text_merge_mode == "ch140_fusion":
        hard_rules.insert(1, "primary_text はASRタイミング上で字幕語彙により訂正済みの融合本文であり、canonicalな文言として扱う。captions / asr フィールドは補助証拠とする。")
    else:
        hard_rules.insert(1, "字幕が利用可能な場合は字幕本文をcanonicalな文言として扱い、ASRはタイミング・音声の補助または字幕欠落区間に限って使う。")
    if boundary_hints:
        hard_rules.append("topic_boundary_hints は本文のLLM解析によるトピック境界候補である。原則としてこれを採用し、CM区間や視覚証拠と明確に矛盾する場合のみ調整する。")
    return {
        "program": {
            "program_id": program_id,
            "source_ts_path": str(input_path),
            "duration_sec": duration_sec,
            "channel_name": channel_name,
            "program_name": program_name,
            "broadcast_start": broadcast_start,
            "broadcast_end": broadcast_end,
        },
        "hard_rules": hard_rules,
        "topic_boundary_hints": boundary_hints or [],
        "deterministic_non_editorial_segments": [seg.model_dump(mode="json") for seg in non_editorial_segments],
        "evidence_windows": [asdict(window) for window in selected_windows],
        "omitted_evidence_window_count": omitted,
        "pipeline_warnings": warnings,
    }


def synthesize_metadata_openai(
    args: argparse.Namespace,
    *,
    input_path: Path,
    duration_sec: float,
    program_id: str,
    channel_name: str | None,
    program_name: str | None,
    broadcast_start: str | None,
    broadcast_end: str | None,
    windows: list[EvidenceWindow],
    deterministic_non_editorial: list[NonEditorialSegment],
    warnings: list[str],
    text_merge_mode: str = "caption_priority",
    boundary_hints: list[dict] | None = None,
) -> ProgramMetadata:
    prompt_path = args.finalizer_prompt or FINALIZER_PROMPT_PATH
    prompt = prompt_path.read_text(encoding="utf-8")
    payload = finalizer_input_payload(
        input_path=input_path,
        duration_sec=duration_sec,
        program_id=program_id,
        channel_name=channel_name,
        program_name=program_name,
        broadcast_start=broadcast_start,
        broadcast_end=broadcast_end,
        windows=windows,
        non_editorial_segments=deterministic_non_editorial,
        warnings=warnings,
        max_windows=args.finalizer_max_windows,
        text_merge_mode=text_merge_mode,
        boundary_hints=boundary_hints,
    )
    user_content = json.dumps(payload, ensure_ascii=False, indent=2)
    status(f"  finalizer model: {args.finalizer_model}")
    status(f"  finalizer evidence windows sent: {min(len(windows), args.finalizer_max_windows)} / {len(windows)}")
    if payload["omitted_evidence_window_count"] > 0:
        status(
            f"  warning: {payload['omitted_evidence_window_count']} trailing evidence windows "
            f"omitted by --finalizer-max-windows={args.finalizer_max_windows}; "
            "topics in the omitted tail may be missing"
        )
    response = openai_client().beta.chat.completions.parse(
        model=args.finalizer_model,
        max_completion_tokens=args.finalizer_max_completion_tokens,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=ProgramMetadata,
    )
    message = response.choices[0].message
    if message.refusal:
        raise RuntimeError(f"finalizer refused: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("finalizer returned no parsed metadata")
    return message.parsed


# A topic is removed only when non-editorial (CM/promo) segments cover at least
# this fraction of its span. Coverage preserves editorial content that spans a
# short interruption while dropping predictions located within non-editorial blocks.
DEFAULT_NON_EDITORIAL_REMOVAL_COVERAGE = 0.7


def non_editorial_coverage_fraction(topic: Topic, non_editorial_segments: list[NonEditorialSegment]) -> float:
    """Fraction of the topic's span covered by the union of non-editorial segments."""
    duration = topic.end_sec - topic.start_sec
    if duration <= 0:
        return 0.0
    # Clip each segment to the topic, then union the intervals so overlapping
    # CM detections are not double-counted.
    clipped = sorted(
        (max(seg.start_sec, topic.start_sec), min(seg.end_sec, topic.end_sec))
        for seg in non_editorial_segments
        if span_overlap(seg.start_sec, seg.end_sec, topic.start_sec, topic.end_sec) > 0
    )
    covered = 0.0
    cur_start: float | None = None
    cur_end = 0.0
    for start, end in clipped:
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            covered += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_start is not None:
        covered += cur_end - cur_start
    return covered / duration


def normalize_final_metadata(
    metadata: ProgramMetadata,
    *,
    input_path: Path,
    program_id: str,
    channel_name: str | None,
    program_name: str | None,
    broadcast_start: str | None,
    broadcast_end: str | None,
    deterministic_non_editorial: list[NonEditorialSegment],
    warnings: list[str],
    duration_sec: float | None = None,
    non_editorial_removal_coverage: float = DEFAULT_NON_EDITORIAL_REMOVAL_COVERAGE,
) -> ProgramMetadata:
    payload = metadata.model_dump(mode="json")
    payload.update({
        "program_id": program_id,
        "source_ts_path": str(input_path),
        "channel_name": channel_name,
        "program_name": program_name,
        "broadcast_start": broadcast_start,
        "broadcast_end": broadcast_end,
    })

    if duration_sec is not None:
        clipped_topics: list[dict[str, Any]] = []
        clipped_items = 0
        for raw_topic in payload.get("topics", []):
            topic = dict(raw_topic)
            topic["start_sec"] = max(0.0, min(float(topic["start_sec"]), duration_sec))
            topic["end_sec"] = max(0.0, min(float(topic["end_sec"]), duration_sec))
            if topic["end_sec"] <= topic["start_sec"]:
                clipped_items += 1
                continue
            clipped_topics.append(topic)
        payload["topics"] = clipped_topics

        clipped_non_editorial: list[dict[str, Any]] = []
        for raw_segment in payload.get("non_editorial_segments", []):
            segment = dict(raw_segment)
            segment["start_sec"] = max(0.0, min(float(segment["start_sec"]), duration_sec))
            segment["end_sec"] = max(0.0, min(float(segment["end_sec"]), duration_sec))
            if segment["end_sec"] <= segment["start_sec"]:
                clipped_items += 1
                continue
            clipped_non_editorial.append(segment)
        payload["non_editorial_segments"] = clipped_non_editorial
        if clipped_items:
            warnings.append(
                f"Removed {clipped_items} item(s) outside the program duration after clipping."
            )

    existing_ne = [NonEditorialSegment.model_validate(item) for item in payload.get("non_editorial_segments", [])]
    merged_ne = existing_ne[:]
    for candidate in deterministic_non_editorial:
        overlaps = [
            span_overlap(candidate.start_sec, candidate.end_sec, item.start_sec, item.end_sec)
            for item in merged_ne
        ]
        if not overlaps or max(overlaps) < 1.0:
            merged_ne.append(candidate)
    merged_ne = sorted(merged_ne, key=lambda item: (item.start_sec, item.end_sec, item.segment_id))

    topics = [Topic.model_validate(item) for item in payload.get("topics", [])]
    kept_topics: list[Topic] = []
    removed_desc: list[str] = []
    for topic in topics:
        coverage = non_editorial_coverage_fraction(topic, merged_ne)
        if coverage >= non_editorial_removal_coverage:
            removed_desc.append(
                f"{topic.topic_id}[{format_seconds(topic.start_sec)}-{format_seconds(topic.end_sec)}]"
                f"({coverage:.0%} non-editorial)"
            )
            continue
        kept_topics.append(topic)
    if removed_desc:
        warnings.append(
            f"Removed {len(removed_desc)} topic(s) mostly covered by deterministic "
            f"non-editorial intervals (>={non_editorial_removal_coverage:.0%}): "
            + ", ".join(removed_desc)
        )

    payload["topics"] = [topic.model_dump(mode="json") for topic in kept_topics]
    payload["non_editorial_segments"] = [seg.model_dump(mode="json") for seg in merged_ne]
    payload["warnings"] = dedupe([*payload.get("warnings", []), *warnings])
    return ProgramMetadata.model_validate(payload)



# File I/O, hashing, manifests, and media probes for resumption and reproducibility.

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_run_artifact_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


def summary_artifact_path(out_dir: Path, *keys: str) -> Path | None:
    summary_path = out_dir / "production_run_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    artifacts = summary.get("artifacts", {}) if isinstance(summary, dict) else {}
    for key in keys:
        direct = resolve_run_artifact_path(summary.get(key) if isinstance(summary, dict) else None)
        if direct:
            return direct
        artifact = resolve_run_artifact_path(artifacts.get(key) if isinstance(artifacts, dict) else None)
        if artifact:
            return artifact
    return None


def copy_json_for_resume(src: Path, dst: Path, label: str) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return dst
    shutil.copy2(src, dst)
    status(f"  copied {label} into run directory for resume: {dst}")
    return dst


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def ffprobe_json(input_path: Path) -> dict[str, Any]:
    proc = run_cmd([
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ])
    return json.loads(proc.stdout)


def fps_from_probe(probe: dict[str, Any]) -> float:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        if isinstance(raw, str) and "/" in raw:
            num, den = raw.split("/", 1)
            try:
                return float(num) / float(den) if float(den) else 0.0
            except ValueError:
                return 0.0
    return 0.0


def image_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        return None, None


def setup_file_logging(log_dir: Path, level: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"
    root = logging.getLogger()
    if not any(isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(log_path) for handler in root.handlers):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        file_handler.setLevel(level.upper())
        root.addHandler(file_handler)
    return log_path


def write_manifest(
    path: Path,
    *,
    input_path: Path,
    args: argparse.Namespace,
    duration_sec: float,
    fps: float,
    log_path: Path,
) -> dict[str, Any]:
    config_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {
        "schema": "mvp_v03_style_manifest_v1",
        "pipeline_version": "mvp-v03-style-2026-06-23",
        "final_public_schema": "ProgramMetadata",
        "input_path": str(input_path),
        "source_sha256": args.source_sha256 or file_sha256(input_path),
        "duration_sec": duration_sec,
        "fps": fps,
        "config": config_payload,
        "config_hash": short_hash(config_payload),
        "log_path": str(log_path),
    }
    write_json(path, manifest)
    status(f"  wrote manifest: {path}")
    return manifest


def extract_audio_artifact(input_path: Path, audio_path: Path, *, sample_rate: int = 16000, channels: int = 1) -> Path:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_path.exists() and audio_path.stat().st_size > 0:
        status(f"  reusing audio artifact: {audio_path}")
        return audio_path
    status(f"  extracting audio artifact: {audio_path}")
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
        str(sample_rate),
        "-ac",
        str(channels),
        str(audio_path),
        "-y",
    ])
    return audio_path


# Convert primary text, OCR, and scenes into candidates and intermediate artifacts.

def build_text_units(primary_segments: list[TextSegment], *, has_captions: bool, has_asr: bool) -> list[dict[str, Any]]:
    if has_captions and has_asr:
        source = "subtitle"
        method = "caption_text_preferred_asr_fallback"
        confidence = 0.86
    elif has_captions:
        source = "subtitle"
        method = "caption_text_lag_corrected"
        confidence = 0.78
    elif has_asr:
        source = "asr"
        method = "fallback_asr_only"
        confidence = 0.55
    else:
        source = "merged"
        method = "none"
        confidence = 0.0
    return [
        {
            "text_unit_id": f"tu{idx:06d}",
            "start_sec": segment.start,
            "end_sec": max(segment.end, segment.start + 0.1),
            "text": segment.text,
            "source": source,
            "original_source_ids": [f"{segment.source}:{idx:06d}"],
            "correction_method": method,
            "confidence": confidence,
        }
        for idx, segment in enumerate(primary_segments)
    ]


def classify_cue_text(text: str) -> tuple[str, str, float]:
    if re.search(r"(コマーシャル|CMの後|このあとは|この後)", text):
        return "cm_transition", "strong", 0.78
    if re.search(r"(ニュースを続けます|引き続き|ここからは)", text):
        return "resume_after_cm", "medium", 0.58
    if TEXT_CUE_RE.search(text):
        return "topic_transition", "medium", 0.62
    return "weak_discourse", "weak", 0.30


def build_cues(primary_segments: list[TextSegment]) -> list[CueRecord]:
    cues: list[CueRecord] = []
    for idx, segment in enumerate(primary_segments):
        cue_type, strength, confidence = classify_cue_text(segment.text)
        if cue_type == "weak_discourse" and not re.search(r"(一方|また|さて|では)", segment.text):
            continue
        cues.append(CueRecord(
            cue_id=f"cue{len(cues):05d}",
            start_sec=segment.start,
            end_sec=segment.end,
            text=truncate_text(segment.text, 180),
            cue_type=cue_type,
            strength=strength,
            confidence=confidence,
        ))
    return cues


def write_text_stage_artifacts(
    *,
    subtitles_dir: Path,
    root_out_dir: Path,
    primary_segments: list[TextSegment],
    caption_segments: list[TextSegment],
    asr_segments: list[TextSegment],
) -> tuple[Path, Path, list[CueRecord]]:
    text_units = build_text_units(primary_segments, has_captions=bool(caption_segments), has_asr=bool(asr_segments))
    corrected_subtitles = [
        {
            "subtitle_id": f"sub{idx:06d}",
            "start_sec": segment.start,
            "end_sec": segment.end,
            "text": segment.text,
            "speaker": segment.speaker,
            "correction_method": "fixed_offset",
        }
        for idx, segment in enumerate(caption_segments)
    ]
    cues = build_cues(primary_segments)
    corrected_path = subtitles_dir / "corrected_subtitles.json"
    text_units_path = root_out_dir / "aligned_text_units.json"
    cues_path = root_out_dir / "cues.json"
    write_json(corrected_path, {"schema": "corrected_subtitles_v1", "segments": corrected_subtitles})
    write_json(text_units_path, {"schema": "aligned_text_units_v1", "text_units": text_units})
    write_json(cues_path, {"schema": "cues_v1", "cues": [asdict(cue) for cue in cues]})
    status(f"  wrote text units: {text_units_path}")
    status(f"  wrote cues: {cues_path} ({len(cues)} cues)")
    return text_units_path, cues_path, cues


def load_ocr_boundaries(path: Path) -> tuple[list[float], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    boundaries = sorted({round(float(value), 2) for value in data.get("boundaries_sec", [])})
    warnings = [str(item) for item in data.get("warnings", [])]
    return boundaries, warnings


def run_telop_ocr_boundaries(args: argparse.Namespace, input_path: Path, ocr_dir: Path) -> tuple[list[float], Path | None, list[str]]:
    warnings: list[str] = []
    samples_path = ocr_dir / "rapidocr_samples.jsonl"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    if not samples_path.exists():
        samples_path.write_text("", encoding="utf-8")
    out_path = ocr_dir / "text_change_candidates.json"
    if args.ocr_mode == "skip":
        status("  RapidOCR sampling skipped")
        write_json(out_path, {"schema": "ocr_text_change_candidates_v1", "boundaries_sec": [], "warnings": ["OCR skipped"]})
        return [], out_path, warnings
    if args.resume and out_path.exists():
        boundaries, cached_warnings = load_ocr_boundaries(out_path)
        if "OCR skipped" not in cached_warnings:
            status(f"  resume: reusing OCR text-change candidates from {out_path} ({len(boundaries)} boundaries)")
            return boundaries, out_path, warnings
        status("  resume: cached OCR was from a skipped run; recomputing")
    if args.logical_start_sec is not None:
        raise RuntimeError("logical program window requires a pre-shifted OCR cache")

    ocr_python = PROJECT_ROOT / ".venv-ocr" / "bin" / "python"
    helper = PROJECT_ROOT / "scripts" / "telop_ocr_broadcast.py"
    if not ocr_python.exists() or not helper.exists():
        message = "RapidOCR helper or .venv-ocr is unavailable"
        if args.ocr_mode == "rapidocr":
            raise RuntimeError(message)
        warnings.append(message)
        status(f"  warning: {message}; OCR boundary candidates skipped")
        write_json(out_path, {"schema": "ocr_text_change_candidates_v1", "boundaries_sec": [], "warnings": warnings})
        return [], out_path, warnings

    try:
        status(f"  running RapidOCR telop sampling: fps={args.ocr_fps}")
        subprocess.run([
            str(ocr_python),
            str(helper),
            "--video",
            str(input_path),
            "--out",
            str(out_path),
            "--fps",
            str(args.ocr_fps),
            "--threshold",
            str(args.ocr_threshold),
            "--merge-gap",
            str(args.ocr_merge_gap),
        ], check=True)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        boundaries = sorted({round(float(value), 2) for value in data.get("boundaries_sec", [])})
        status(f"  OCR text-change boundaries: {len(boundaries)}")
        return boundaries, out_path, warnings
    except Exception as exc:
        message = f"RapidOCR sampling failed: {exc}"
        if args.ocr_mode == "rapidocr":
            raise RuntimeError(message) from exc
        warnings.append(message)
        status(f"  warning: {message}")
        write_json(out_path, {"schema": "ocr_text_change_candidates_v1", "boundaries_sec": [], "warnings": warnings})
        return [], out_path, warnings


def normalize_scene_ranges_for_program(
    scene_ranges: list[tuple[float, float]],
    duration_sec: float,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    raw = [(float(start), float(end)) for start, end in scene_ranges if float(end) > float(start)]
    if not raw:
        return [(0.0, round(duration_sec, 2))], {"normalized": False, "reason": "empty_scene_ranges"}

    min_start = min(start for start, _end in raw)
    max_end = max(end for _start, end in raw)
    needs_offset = min_start > duration_sec or max_end > duration_sec + 5.0
    offset = raw[0][0] if needs_offset else 0.0

    normalized: list[tuple[float, float]] = []
    for start, end in raw:
        rel_start = max(0.0, start - offset)
        rel_end = max(0.0, end - offset)
        rel_start = min(rel_start, duration_sec)
        rel_end = min(rel_end, duration_sec)
        if rel_end > rel_start:
            normalized.append((round(rel_start, 2), round(rel_end, 2)))

    if not normalized:
        normalized = [(0.0, round(duration_sec, 2))]

    return normalized, {
        "normalized": needs_offset,
        "offset_sec": round(offset, 6),
        "raw_min_start_sec": round(min_start, 2),
        "raw_max_end_sec": round(max_end, 2),
        "n_after_normalization": len(normalized),
    }


def clip_scene_ranges_to_logical_window(
    scene_ranges: list[tuple[float, float]],
    *,
    logical_start_sec: float,
    logical_end_sec: float,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Clip source-relative scene ranges and shift them to logical-program time.

    Scene detection and refinement still inspect the original, untrimmed media.
    Only ranges intersecting the manually fixed program window survive, and the
    public/output timeline remains zero based.
    """
    duration_sec = logical_end_sec - logical_start_sec
    clipped: list[tuple[float, float]] = []
    for raw_start, raw_end in scene_ranges:
        source_start = max(float(raw_start), logical_start_sec)
        source_end = min(float(raw_end), logical_end_sec)
        if source_end <= source_start:
            continue
        clipped.append((
            round(max(0.0, source_start - logical_start_sec), 2),
            round(min(duration_sec, source_end - logical_start_sec), 2),
        ))

    if not clipped:
        clipped = [(0.0, round(duration_sec, 2))]
    return clipped, {
        "normalized": True,
        "method": "clip_and_shift_logical_window",
        "offset_sec": round(logical_start_sec, 6),
        "logical_start_sec": round(logical_start_sec, 6),
        "logical_end_sec": round(logical_end_sec, 6),
        "n_source_ranges": len(scene_ranges),
        "n_after_normalization": len(clipped),
    }


def scene_candidates_from_ranges(scene_ranges: list[tuple[float, float]], duration_sec: float, source: str) -> list[BoundaryCandidateRecord]:
    if not scene_ranges:
        cuts = [0.0, round(duration_sec, 2)]
    else:
        cuts = [round(scene_ranges[0][0], 2)] + [round(end, 2) for _start, end in scene_ranges]
    candidates: list[BoundaryCandidateRecord] = []
    for idx, cut in enumerate(sorted(set(cuts))):
        candidates.append(BoundaryCandidateRecord(
            boundary_id=f"b_{source}_{idx:06d}",
            time_sec=cut,
            boundary_type="scene",
            source=source,
            score=0.72 if 0.0 < cut < duration_sec else 1.0,
            reasons=["resumed scene boundary" if source == "resume" else "PySceneDetect HashDetector boundary" if 0.0 < cut < duration_sec else "program edge"],
        ))
    return candidates


def scene_records_from_ranges(scene_ranges: list[tuple[float, float]], existing_rows: list[dict[str, Any]] | None = None) -> list[SceneRecord]:
    scenes: list[SceneRecord] = []
    existing_rows = existing_rows or []
    for idx, (start, end) in enumerate(scene_ranges):
        scene_id = str(existing_rows[idx].get("scene_id") if idx < len(existing_rows) else f"scene_{idx:06d}")
        scenes.append(SceneRecord(
            scene_id=scene_id,
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            start_boundary_id=str(existing_rows[idx].get("start_boundary_id") if idx < len(existing_rows) else f"b_scene_{idx:06d}_start"),
            end_boundary_id=str(existing_rows[idx].get("end_boundary_id") if idx < len(existing_rows) else f"b_scene_{idx:06d}_end"),
            representative_frame_ids=list(existing_rows[idx].get("representative_frame_ids", [])) if idx < len(existing_rows) else [],
        ))
    return scenes


def load_scene_artifacts_for_resume(scenes_dir: Path, duration_sec: float, *, expect_refine: bool = False) -> tuple[list[SceneRecord], list[BoundaryCandidateRecord], dict[str, Any]] | None:
    fused_path = scenes_dir / "fused_scenes.json"
    if not fused_path.exists():
        return None
    try:
        payload = json.loads(fused_path.read_text(encoding="utf-8"))
        cached_refine = bool(dict(payload.get("stats") or {}).get("scene_refine"))
        if cached_refine != expect_refine:
            status(
                "  resume: cached scenes were computed with "
                f"--scene-refine={'on' if cached_refine else 'off'} but this run wants "
                f"{'on' if expect_refine else 'off'}; recomputing scenes"
            )
            return None
        rows = payload.get("scenes", [])
        scene_ranges = [(float(row["start_sec"]), float(row["end_sec"])) for row in rows]
        scene_ranges, info = normalize_scene_ranges_for_program(scene_ranges, duration_sec)
        scenes = scene_records_from_ranges(scene_ranges, rows)
        candidates = scene_candidates_from_ranges(scene_ranges, duration_sec, "resume")
        stats = dict(payload.get("stats", {}))
        stats.update({
            "resumed": True,
            "resume_path": str(fused_path),
            "pipeline_scene_time_normalization": info,
            "n_scenes": len(scenes),
        })
        if info.get("normalized"):
            for idx, scene in enumerate(scenes):
                scenes[idx] = SceneRecord(
                    scene.scene_id,
                    scene.start_sec,
                    scene.end_sec,
                    scene.start_boundary_id,
                    scene.end_boundary_id,
                    [],
                )
            status(
                "  resume: normalized cached scene timestamps by "
                f"-{info['offset_sec']:.2f}s to program-relative seconds"
            )
        write_scene_artifacts(scenes_dir, scenes, candidates, stats)
        status(f"  resume: reusing scenes from {fused_path} ({len(scenes)} scenes)")
        return scenes, candidates, stats
    except Exception as exc:
        status(f"  resume: cached scenes were not usable ({exc}); recomputing")
        return None


def detect_scene_records(
    args: argparse.Namespace,
    input_path: Path,
    scenes_dir: Path,
    duration_sec: float,
    ocr_boundaries_sec: list[float],
) -> tuple[list[SceneRecord], list[BoundaryCandidateRecord], dict[str, Any], list[str]]:
    warnings: list[str] = []
    if args.scene_mode == "skip":
        status("  scene detection skipped; using one full-program scene")
        scenes = [SceneRecord("scene_000000", 0.0, round(duration_sec, 2), "program_start", "program_end", [])]
        stats = {"strategy": "skip", "duration_sec": duration_sec, "n_scenes": 1}
        candidates = [
            BoundaryCandidateRecord("b_program_start", 0.0, "scene", "manual", 1.0, ["program start"]),
            BoundaryCandidateRecord("b_program_end", round(duration_sec, 2), "scene", "manual", 1.0, ["program end"]),
        ]
        write_scene_artifacts(scenes_dir, scenes, candidates, stats)
        return scenes, candidates, stats, warnings
    if args.resume:
        resumed = load_scene_artifacts_for_resume(
            scenes_dir, duration_sec, expect_refine=bool(getattr(args, "scene_refine", False))
        )
        if resumed:
            scenes, candidates, stats = resumed
            return scenes, candidates, stats, warnings
    try:
        from src.extract_frames import detect_scenes

        status(f"  detecting scenes: strategy={args.scene_strategy} detector=hash")
        scene_ranges, _origin, stats = detect_scenes(
            input_path,
            strategy=args.scene_strategy,
            detector="hash",
            threshold=args.scene_threshold,
            min_scene_len=args.scene_min_len_frames,
            # An empty list is meaningful: OCR was explicitly skipped or ran
            # successfully with zero changes. Converting it to None would make
            # option_d launch the dense RapidOCR helper as a hidden fallback.
            ocr_boundaries_sec=ocr_boundaries_sec,
        )
        logical_window = args.logical_start_sec is not None
        if logical_window:
            assert args.logical_end_sec is not None
            # Refinement reads the original media and therefore must receive
            # source-relative times. Clip now, shift to zero only after refine.
            source_scene_ranges = [
                (
                    max(float(start), args.logical_start_sec),
                    min(float(end), args.logical_end_sec),
                )
                for start, end in scene_ranges
                if min(float(end), args.logical_end_sec)
                > max(float(start), args.logical_start_sec)
            ]
            if not source_scene_ranges:
                source_scene_ranges = [(args.logical_start_sec, args.logical_end_sec)]
            scene_ranges = source_scene_ranges
        else:
            scene_ranges, scene_time_info = normalize_scene_ranges_for_program(
                scene_ranges, duration_sec
            )
            stats["pipeline_scene_time_normalization"] = scene_time_info
            if scene_time_info.get("normalized"):
                status(
                    "  normalized scene timestamps by "
                    f"-{scene_time_info['offset_sec']:.2f}s to program-relative seconds"
                )
        if getattr(args, "scene_refine", False):
            from src.scene_refine import refine_scene_ranges

            status("  scene refine: second-pass ContentDetector + silence snap + forced split")
            audio_path = scenes_dir.parent / "media" / "audio.wav"
            scene_ranges, refine_stats = refine_scene_ranges(
                input_path,
                scene_ranges,
                silence_media_path=audio_path if audio_path.exists() else None,
            )
            stats["scene_refine"] = refine_stats
            status(
                f"  scene refine: {refine_stats['n_scenes_in']} -> {refine_stats['n_scenes_out']} scenes "
                f"(+{refine_stats['n_cuts_inserted']} cuts, snapped {refine_stats['n_cuts_snapped']}, "
                f"merged {refine_stats['n_merged']}, forced {refine_stats['n_forced_split_cuts']})"
            )
        if logical_window:
            assert args.logical_start_sec is not None
            assert args.logical_end_sec is not None
            scene_ranges, scene_time_info = clip_scene_ranges_to_logical_window(
                scene_ranges,
                logical_start_sec=args.logical_start_sec,
                logical_end_sec=args.logical_end_sec,
            )
            stats["pipeline_scene_time_normalization"] = scene_time_info
            status(
                "  clipped source scenes to logical window and normalized by "
                f"-{args.logical_start_sec:.2f}s"
            )
        scenes: list[SceneRecord] = []
        scenes = scene_records_from_ranges(scene_ranges)
        candidates = scene_candidates_from_ranges(scene_ranges, duration_sec, "hash")
        stats["n_scenes"] = len(scenes)
        write_scene_artifacts(scenes_dir, scenes, candidates, stats)
        return scenes, candidates, stats, warnings
    except Exception as exc:
        message = f"scene detection failed: {exc}"
        if not args.allow_partial:
            raise RuntimeError(message) from exc
        warnings.append(message)
        status(f"  warning: {message}; using one full-program scene")
        scenes = [SceneRecord("scene_000000", 0.0, round(duration_sec, 2), "program_start", "program_end", [])]
        candidates = [BoundaryCandidateRecord("b_scene_fallback", 0.0, "scene", "manual", 0.3, [message])]
        stats = {"strategy": "fallback", "duration_sec": duration_sec, "n_scenes": 1, "error": message}
        write_scene_artifacts(scenes_dir, scenes, candidates, stats)
        return scenes, candidates, stats, warnings


def write_scene_artifacts(
    scenes_dir: Path,
    scenes: list[SceneRecord],
    candidates: list[BoundaryCandidateRecord],
    stats: dict[str, Any],
) -> None:
    write_json(scenes_dir / "hash_candidates.json", {
        "schema": "hash_scene_candidates_v1",
        "stats": stats,
        "candidates": [asdict(candidate) for candidate in candidates if candidate.source == "hash" or candidate.source == "manual"],
    })
    write_json(scenes_dir / "fused_scenes.json", {
        "schema": "fused_scenes_v1",
        "stats": stats,
        "scenes": [asdict(scene) for scene in scenes],
    })
    status(f"  wrote scenes: {scenes_dir / 'fused_scenes.json'} ({len(scenes)} scenes)")


# Produce representative frames, combined boundaries, and auxiliary exports.

def extract_single_frame_jpeg(input_path: Path, out_path: Path, *, timestamp_sec: float, max_dimension: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    vf = f"scale='if(gt(iw,ih),{max_dimension},-2)':'if(gt(ih,iw),{max_dimension},-2)'"
    commands = [
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(round(timestamp_sec, 2)),
            "-i",
            str(input_path),
            "-vframes",
            "1",
            "-vf",
            vf,
            "-q:v",
            "2",
            str(out_path),
            "-y",
        ],
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-ss",
            str(round(timestamp_sec, 2)),
            "-vframes",
            "1",
            "-vf",
            vf,
            "-q:v",
            "2",
            str(out_path),
            "-y",
        ],
    ]
    errors: list[str] = []
    for cmd in commands:
        try:
            run_cmd(cmd)
        except subprocess.CalledProcessError as exc:
            errors.append(str(exc))
        if out_path.exists() and out_path.stat().st_size > 0:
            return
        if out_path.exists():
            out_path.unlink()
    details = "; ".join(errors[-2:]) if errors else "ffmpeg exited without writing an image"
    raise RuntimeError(f"failed to extract representative frame at {timestamp_sec:.2f}s to {out_path}: {details}")


def representative_frame_timestamp(start_sec: float, end_sec: float, *, media_duration_sec: float) -> float:
    duration = max(end_sec - start_sec, 0.1)
    ts = start_sec + duration / 2.0
    if duration > 1.0:
        ts = min(max(start_sec + 0.25, ts), end_sec - 0.25)
    # The 0.1s duration floor can push the midpoint of a tiny tail scene past
    # the last decodable frame, where ffmpeg writes no image. Keep the seek
    # inside the scene and clear of the media tail; the scene start itself is
    # always a decodable position.
    ts = min(ts, end_sec)
    if media_duration_sec > 0:
        ts = min(ts, media_duration_sec - 0.5)
    return max(ts, start_sec)


def extract_scene_representative_frames(
    input_path: Path,
    scenes: list[SceneRecord],
    frames_dir: Path,
    *,
    media_duration_sec: float,
    max_frames: int | None,
    max_dimension: int,
    resume: bool = False,
    source_time_offset_sec: float = 0.0,
) -> tuple[list[FrameRecord], list[SceneRecord]]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    selected_scenes = scenes if max_frames is None else scenes[:max_frames]
    frames: list[FrameRecord] = []
    scene_updates: dict[str, list[str]] = {scene.scene_id: [] for scene in scenes}
    status(f"  extracting {len(selected_scenes)} scene representative frames to {frames_dir}")
    for idx, scene in enumerate(selected_scenes):
        ts = representative_frame_timestamp(
            scene.start_sec, scene.end_sec, media_duration_sec=media_duration_sec
        )
        source_ts = ts + source_time_offset_sec
        frame_id = f"{scene.scene_id}_mid"
        out_path = frames_dir / f"{frame_id}.jpg"
        if resume and out_path.exists() and out_path.stat().st_size > 0:
            progress("scene-frames", idx + 1, len(selected_scenes), f"reuse {scene.scene_id} {format_seconds(ts)}")
        else:
            progress("scene-frames", idx + 1, len(selected_scenes), f"{scene.scene_id} {format_seconds(ts)}")
            extract_single_frame_jpeg(
                input_path,
                out_path,
                timestamp_sec=source_ts,
                max_dimension=max_dimension,
            )
        frames.append(FrameRecord(frame_id=frame_id, frame_index=idx, timestamp_sec=round(ts, 2), path=out_path))
        scene_updates[scene.scene_id].append(frame_id)
    updated = [
        SceneRecord(scene.scene_id, scene.start_sec, scene.end_sec, scene.start_boundary_id, scene.end_boundary_id, scene_updates.get(scene.scene_id, []))
        for scene in scenes
    ]
    return frames, updated


def build_boundary_candidates(
    *,
    scene_candidates: list[BoundaryCandidateRecord],
    cues: list[CueRecord],
    ocr_boundaries_sec: list[float],
    non_editorial: list[tuple[float, float, str, list[str]]],
) -> list[BoundaryCandidateRecord]:
    candidates = list(scene_candidates)
    for cue in cues:
        candidates.append(BoundaryCandidateRecord(
            boundary_id=f"b_{cue.cue_id}",
            time_sec=round(cue.start_sec, 2),
            boundary_type="topic" if cue.cue_type in {"topic_transition", "resume_after_cm", "breaking_news"} else "unknown",
            source="cue",
            score=cue.confidence,
            reasons=[cue.text],
        ))
    for idx, boundary in enumerate(ocr_boundaries_sec):
        candidates.append(BoundaryCandidateRecord(
            boundary_id=f"b_ocr_{idx:06d}",
            time_sec=boundary,
            boundary_type="scene",
            source="ocr_text_change",
            score=0.55,
            reasons=["RapidOCR text change candidate"],
        ))
    for idx, (start, end, kind, _frames) in enumerate(non_editorial):
        candidates.append(BoundaryCandidateRecord(f"b_cm_{idx:06d}_start", round(start, 2), "cm_start", "cm_detector", 0.86, [kind]))
        candidates.append(BoundaryCandidateRecord(f"b_cm_{idx:06d}_end", round(end, 2), "cm_end", "cm_detector", 0.86, [kind]))
    return sorted(candidates, key=lambda item: (item.time_sec, item.boundary_id))


def write_boundary_and_cm_artifacts(
    *,
    scenes_dir: Path,
    cm_dir: Path,
    candidates: list[BoundaryCandidateRecord],
    non_editorial: list[tuple[float, float, str, list[str]]],
) -> tuple[Path, Path]:
    boundary_path = scenes_dir / "boundary_candidates.json"
    cm_path = cm_dir / "cm_candidates.json"
    write_json(boundary_path, {
        "schema": "boundary_candidates_v1",
        "candidates": [asdict(candidate) for candidate in candidates],
    })
    write_json(cm_path, {
        "schema": "cm_candidates_v1",
        "candidates": [
            {
                "cm_candidate_id": f"cm{idx:05d}",
                "start_sec": start,
                "end_sec": end,
                "kind": kind,
                "source": "vision_llm_or_fallback",
                "representative_frame_ids": frame_ids,
            }
            for idx, (start, end, kind, frame_ids) in enumerate(non_editorial)
        ],
    })
    status(f"  wrote boundary candidates: {boundary_path}")
    status(f"  wrote CM candidates: {cm_path}")
    return boundary_path, cm_path


def write_frames_manifest(path: Path, frames: list[FrameRecord], scenes: list[SceneRecord], metadata: ProgramMetadata) -> None:
    # Topic no longer carries representative frame IDs, so per-frame topic
    # attribution now comes only from non-editorial segments.
    topic_by_frame: dict[str, list[str]] = {}
    scene_by_frame = {frame_id: scene.scene_id for scene in scenes for frame_id in scene.representative_frame_ids}
    rows = []
    for frame in frames:
        width, height = image_dimensions(frame.path)
        rows.append({
            "frame_id": frame.frame_id,
            "scene_id": scene_by_frame.get(frame.frame_id, frame.frame_id.rsplit("_", 1)[0]),
            "topic_ids": dedupe(topic_by_frame.get(frame.frame_id, [])),
            "path": str(frame.path),
            "time_sec": frame.timestamp_sec,
            "sha256": file_sha256(frame.path) if frame.path.exists() else None,
            "width": width,
            "height": height,
            "selection_reason": "midpoint",
        })
    write_json(path, {"schema": "frames_manifest_v1", "frames": rows})
    status(f"  wrote frames manifest: {path}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def export_search_and_research(
    *,
    output_dir: Path,
    metadata: ProgramMetadata,
    scenes: list[SceneRecord],
    annotation_records: list[FrameAnnotationRecord],
) -> dict[str, str]:
    search_rows: list[dict[str, Any]] = []
    research_rows: list[dict[str, Any]] = []
    scene_frame_paths = {record.frame.frame_id: str(record.frame.path) for record in annotation_records}
    for topic in metadata.topics:
        # Topic no longer carries representative frame IDs in the public schema.
        frame_ids: list[str] = []
        frame_paths: list[str] = []
        text_excerpt = " ".join(topic.evidence_texts[:8])
        search_rows.append({
            "document_id": f"{metadata.program_id}:{topic.topic_id}",
            "program_id": metadata.program_id,
            "topic_id": topic.topic_id,
            "title_ja": topic.title,
            "summary_ja": topic.summary,
            "category": topic.domain,
            "keywords_ja": [],
            "named_entities_ja": [],
            "time_ranges": [{"start_sec": topic.start_sec, "end_sec": topic.end_sec}],
            "representative_frame_paths": frame_paths,
            "text_excerpt_ja": truncate_text(text_excerpt, 600),
            "source_station_name": metadata.channel_name,
            "program_title": metadata.program_name,
            "recorded_start_at": metadata.broadcast_start,
            "warnings": [topic.notes] if topic.notes else [],
        })
        research_rows.append({
            "record_id": f"{metadata.program_id}:{topic.topic_id}",
            "program_id": metadata.program_id,
            "level": "topic",
            "item_id": topic.topic_id,
            "parent_ids": [],
            "start_sec": topic.start_sec,
            "end_sec": topic.end_sec,
            "label": topic.domain,
            "title_ja": topic.title,
            "summary_ja": topic.summary,
            "text_ja": truncate_text(text_excerpt, 1000),
            "visible_telops": [],
            "representative_frame_paths": frame_paths,
            "evidence_ids": frame_ids,
            "warnings": [topic.notes] if topic.notes else [],
        })
    for scene in scenes:
        research_rows.append({
            "record_id": f"{metadata.program_id}:{scene.scene_id}",
            "program_id": metadata.program_id,
            "level": "scene",
            "item_id": scene.scene_id,
            "parent_ids": [],
            "start_sec": scene.start_sec,
            "end_sec": scene.end_sec,
            "label": "scene",
            "title_ja": None,
            "summary_ja": None,
            "text_ja": "",
            "visible_telops": [],
            "representative_frame_paths": [scene_frame_paths[fid] for fid in scene.representative_frame_ids if fid in scene_frame_paths],
            "evidence_ids": scene.representative_frame_ids,
            "warnings": [],
        })
    for segment in metadata.non_editorial_segments:
        research_rows.append({
            "record_id": f"{metadata.program_id}:{segment.segment_id}",
            "program_id": metadata.program_id,
            "level": "non_editorial",
            "item_id": segment.segment_id,
            "parent_ids": [],
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "label": segment.non_editorial_kind,
            "title_ja": None,
            "summary_ja": segment.summary,
            "text_ja": segment.summary or "",
            "visible_telops": [],
            "representative_frame_paths": [scene_frame_paths[fid] for fid in segment.representative_frame_ids if fid in scene_frame_paths],
            "evidence_ids": segment.representative_frame_ids,
            "warnings": [segment.notes] if segment.notes else [],
        })

    search_path = output_dir / "search_index_documents.jsonl"
    research_path = output_dir / "research_dataset.jsonl"
    topics_csv = output_dir / "topics.csv"
    scenes_csv = output_dir / "scenes.csv"
    write_jsonl(search_path, search_rows)
    write_jsonl(research_path, research_rows)
    write_csv(topics_csv, [row for row in research_rows if row["level"] == "topic"], ["record_id", "program_id", "level", "item_id", "start_sec", "end_sec", "label", "title_ja", "summary_ja", "warnings"])
    write_csv(scenes_csv, [row for row in research_rows if row["level"] == "scene"], ["record_id", "program_id", "level", "item_id", "start_sec", "end_sec", "label", "representative_frame_paths", "warnings"])
    status(f"  wrote search export: {search_path}")
    status(f"  wrote research export: {research_path}")
    return {
        "search_index_documents": str(search_path),
        "research_dataset": str(research_path),
        "topics_csv": str(topics_csv),
        "scenes_csv": str(scenes_csv),
    }

def write_metadata(path: Path, metadata: ProgramMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = ProgramMetadata.model_validate(metadata.model_dump(mode="json"))
    path.write_text(json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    status(f"  wrote metadata: {path}")
    status("  validation: ProgramMetadata accepted")


# CLI orchestration: invoke the functions above in ten stages.

def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run all stages and return a summary of artifacts and record counts.

    Stages 1-3 inspect the input and create captions and ASR. Stage 4 builds
    primary text and cues. Stages 5-6 create OCR, scenes, representative frames,
    and visual annotations. Stage 7 aligns evidence into windows and optionally
    splits them using text-boundary hints. Stage 8 invokes the finalizer and
    deterministically normalizes non-editorial intervals, times, and schema.
    Stages 9-10 write public JSON, auxiliary exports, the resume manifest, and
    the run summary. Intermediate artifacts let ``--resume`` skip completed
    expensive stages.
    """
    load_dotenv(args.env_file if args.env_file else None)
    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    media_dir = out_dir / "media"
    subtitles_dir = out_dir / "subtitles"
    asr_dir = out_dir / "asr"
    ocr_dir = out_dir / "ocr"
    scenes_dir = out_dir / "scenes"
    frames_dir = scenes_dir / "frames"
    cm_dir = out_dir / "cm"
    llm_dir = out_dir / "llm"
    output_profile_dir = out_dir / "output"
    logs_dir = out_dir / "logs"
    for directory in (media_dir, subtitles_dir, asr_dir, ocr_dir, scenes_dir, frames_dir, cm_dir, llm_dir, output_profile_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = setup_file_logging(logs_dir, args.log_level)

    program_id = args.program_id or input_path.stem
    warnings: list[str] = []
    started = time.time()

    status("Production TS metadata pipeline")
    status(f"  input:  {input_path}")
    status(f"  out:    {out_dir}")
    status(f"  env:    {args.env_file if args.env_file else '(none)'}")
    status(f"  log:    {log_path}")
    if args.resume:
        status("  resume: enabled; existing stage artifacts in --out-dir will be reused when valid")

    stage(1, 10, "Probe input and manifest")
    probe = ffprobe_json(input_path)
    write_json(media_dir / "probe.json", probe)
    source_duration_sec = float(probe.get("format", {}).get("duration") or ffprobe_duration(input_path))
    duration_sec = logical_program_duration(args, source_duration_sec)
    fps = fps_from_probe(probe)
    if args.logical_start_sec is not None:
        status(
            f"  logical program window: {args.logical_start_sec:.3f}s-"
            f"{args.logical_end_sec:.3f}s -> {duration_sec:.3f}s"
        )
    status(f"  duration: {format_seconds(duration_sec)} ({duration_sec:.2f}s)")
    status(f"  fps: {fps:.3f}")
    if not args.skip_media_audio:
        extract_audio_artifact(input_path, media_dir / "audio.wav")
    manifest = write_manifest(out_dir / "manifest.json", input_path=input_path, args=args, duration_sec=duration_sec, fps=fps, log_path=log_path)

    stage(2, 10, "Captions")
    if args.logical_start_sec is not None and args.caption_mode != "skip" and args.captions_json is None:
        raise RuntimeError("logical program window requires a pre-shifted captions JSON")
    caption_segments, captions_path, caption_warnings = extract_or_load_captions(args, input_path, subtitles_dir)
    warnings.extend(caption_warnings)

    stage(3, 10, "ASR")
    if args.logical_start_sec is not None and args.asr_mode != "skip" and args.asr_json is None:
        raise RuntimeError("logical program window requires a pre-shifted ASR JSON")
    try:
        asr_segments, asr_path = extract_or_load_asr(args, input_path, asr_dir)
    except Exception as exc:
        if not caption_segments:
            raise
        message = f"ASR failed; continuing with captions only: {exc}"
        warnings.append(message)
        status(f"  warning: {message}")
        asr_segments, asr_path = [], None

    stage(4, 10, "Text alignment and cues")
    fused_segments: list[TextSegment] = []
    fusion_path: Path | None = None
    if args.text_merge_mode == "ch140_fusion":
        fused_segments, fusion_path, fusion_warnings = run_ch140_fusion(
            args, asr_segments, caption_segments, out_dir, program_id, duration_sec
        )
        warnings.extend(fusion_warnings)
    primary_segments, primary_warnings = build_primary_segments(
        asr_segments, caption_segments, fused_segments
    )
    warnings.extend(primary_warnings)
    status(f"  primary segments: {len(primary_segments)}")
    status(f"  caption evidence segments: {len(caption_segments)}")
    status(f"  ASR evidence segments: {len(asr_segments)}")
    text_units_path, cues_path, cue_records = write_text_stage_artifacts(
        subtitles_dir=subtitles_dir,
        root_out_dir=out_dir,
        primary_segments=primary_segments,
        caption_segments=caption_segments,
        asr_segments=asr_segments,
    )

    stage(5, 10, "OCR and scene candidates")
    ocr_boundaries_sec, ocr_candidates_path, ocr_warnings = run_telop_ocr_boundaries(args, input_path, ocr_dir)
    warnings.extend(ocr_warnings)
    scene_records, scene_candidates, scene_stats, scene_warnings = detect_scene_records(
        args, input_path, scenes_dir, duration_sec, ocr_boundaries_sec
    )
    warnings.extend(scene_warnings)

    stage(6, 10, "Representative frames and VLM annotations")
    annotations_path = llm_dir / "scene_annotations.json"
    need_frames = not (args.frame_annotations and args.frame_annotations.exists())
    need_frames = need_frames and not ((args.reuse_annotations or args.resume) and annotations_path.exists())
    if need_frames:
        frames, scene_records = extract_scene_representative_frames(
            input_path,
            scene_records,
            frames_dir,
            media_duration_sec=duration_sec,
            max_frames=args.max_frames,
            max_dimension=args.max_frame_dimension,
            resume=args.resume,
            source_time_offset_sec=args.logical_start_sec or 0.0,
        )
        write_scene_artifacts(scenes_dir, scene_records, scene_candidates, scene_stats)
    else:
        frames = []
        status("  reusing frame annotations; frame extraction skipped")
    annotation_records = annotate_or_load_frames(args, frames, duration_sec, annotations_path)

    stage(7, 10, "Evidence fusion and boundary candidates")
    non_editorial = non_editorial_intervals(annotation_records)
    deterministic_non_editorial = build_non_editorial_segments(non_editorial)
    status(f"  non-editorial intervals: {len(non_editorial)}")
    for idx, (start, end, kind, frame_ids) in enumerate(non_editorial, start=1):
        status(f"    [non-editorial {idx}] {format_seconds(start)}-{format_seconds(end)} kind={kind} frames={len(frame_ids)}")
    boundary_candidates = build_boundary_candidates(
        scene_candidates=scene_candidates,
        cues=cue_records,
        ocr_boundaries_sec=ocr_boundaries_sec,
        non_editorial=non_editorial,
    )
    boundary_candidates_path, cm_candidates_path = write_boundary_and_cm_artifacts(
        scenes_dir=scenes_dir,
        cm_dir=cm_dir,
        candidates=boundary_candidates,
        non_editorial=non_editorial,
    )
    boundary_hints: list[dict] = []
    topic_boundaries_path: Path | None = None
    if args.boundary_hints_mode == "llm_text":
        boundary_hints, topic_boundaries_path, hint_warnings = run_topic_boundary_hints(
            args, primary_segments, boundary_candidates, non_editorial, llm_dir
        )
        warnings.extend(hint_warnings)
    evidence_windows = build_evidence_windows(
        duration_sec=duration_sec,
        window_sec=args.evidence_window_sec,
        captions=caption_segments,
        asr=asr_segments,
        primary=primary_segments,
        annotations=annotation_records,
        non_editorial=non_editorial,
        extra_cuts=[h["time_sec"] for h in boundary_hints],
    )
    evidence_path = out_dir / "evidence_windows_mvp.json"
    write_evidence_json(
        evidence_path,
        input_path=input_path,
        duration_sec=duration_sec,
        captions_path=captions_path,
        asr_path=asr_path,
        annotations_path=annotations_path if annotation_records else None,
        windows=evidence_windows,
        non_editorial=non_editorial,
    )

    stage(8, 10, "Final metadata synthesis")
    metadata: ProgramMetadata
    if args.finalizer == "openai":
        try:
            metadata = synthesize_metadata_openai(
                args,
                input_path=input_path,
                duration_sec=duration_sec,
                program_id=program_id,
                channel_name=args.channel_name,
                program_name=args.program_name,
                broadcast_start=args.broadcast_start,
                broadcast_end=args.broadcast_end,
                windows=evidence_windows,
                deterministic_non_editorial=deterministic_non_editorial,
                warnings=warnings,
                text_merge_mode=args.text_merge_mode,
                boundary_hints=boundary_hints,
            )
        except Exception as exc:
            if not args.allow_heuristic_fallback:
                raise
            warning = f"OpenAI finalizer failed; used heuristic fallback: {exc}"
            warnings.append(warning)
            status(f"  warning: {warning}")
            metadata = build_heuristic_metadata(
                input_path=input_path,
                duration_sec=duration_sec,
                program_id=program_id,
                channel_name=args.channel_name,
                program_name=args.program_name,
                broadcast_start=args.broadcast_start,
                broadcast_end=args.broadcast_end,
                primary_segments=primary_segments,
                caption_segments=caption_segments,
                asr_segments=asr_segments,
                annotation_records=annotation_records,
                non_editorial=non_editorial,
                min_topic_duration_sec=args.min_topic_duration_sec,
                initial_warnings=warnings,
            )
    else:
        metadata = build_heuristic_metadata(
            input_path=input_path,
            duration_sec=duration_sec,
            program_id=program_id,
            channel_name=args.channel_name,
            program_name=args.program_name,
            broadcast_start=args.broadcast_start,
            broadcast_end=args.broadcast_end,
            primary_segments=primary_segments,
            caption_segments=caption_segments,
            asr_segments=asr_segments,
            annotation_records=annotation_records,
            non_editorial=non_editorial,
            min_topic_duration_sec=args.min_topic_duration_sec,
            initial_warnings=warnings,
        )

    metadata = normalize_final_metadata(
        metadata,
        input_path=input_path,
        program_id=program_id,
        channel_name=args.channel_name,
        program_name=args.program_name,
        broadcast_start=args.broadcast_start,
        broadcast_end=args.broadcast_end,
        deterministic_non_editorial=deterministic_non_editorial,
        warnings=warnings,
        duration_sec=duration_sec,
    )
    status(f"  topics: {len(metadata.topics)}")
    status(f"  non_editorial_segments: {len(metadata.non_editorial_segments)}")

    stage(9, 10, "Write final output and auxiliary exports")
    metadata_path = (args.output or out_dir / "program_metadata_mvp.json").resolve()
    v03_metadata_path = output_profile_dir / "metadata.final.json"
    write_metadata(metadata_path, metadata)
    write_metadata(v03_metadata_path, metadata)
    export_files = export_search_and_research(
        output_dir=output_profile_dir,
        metadata=metadata,
        scenes=scene_records,
        annotation_records=annotation_records,
    )
    frames_manifest_path = output_profile_dir / "frames_manifest.json"
    manifest_frames = frames or [record.frame for record in annotation_records if record.frame.path]
    write_frames_manifest(frames_manifest_path, manifest_frames, scene_records, metadata)
    diagnostics_path = output_profile_dir / "diagnostics.json"
    write_json(diagnostics_path, {
        "schema": "mvp_v03_style_diagnostics_v1",
        "warnings": metadata.warnings,
        "manifest": manifest,
        "scene_stats": scene_stats,
        "n_boundary_candidates": len(boundary_candidates),
        "n_evidence_windows": len(evidence_windows),
    })

    stage(10, 10, "Write run summary")
    summary_path = out_dir / "production_run_summary.json"
    artifacts = {
        "manifest": str(out_dir / "manifest.json"),
        "probe_json": str(media_dir / "probe.json"),
        "audio_wav": str(media_dir / "audio.wav") if (media_dir / "audio.wav").exists() else None,
        "captions_json": str(captions_path) if captions_path else None,
        "corrected_subtitles_json": str(subtitles_dir / "corrected_subtitles.json"),
        "asr_json": str(asr_path) if asr_path else None,
        "transcript_unified_json": str(fusion_path) if fusion_path else None,
        "topic_boundaries_json": str(topic_boundaries_path) if topic_boundaries_path else None,
        "aligned_text_units_json": str(text_units_path),
        "cues_json": str(cues_path),
        "ocr_text_change_candidates_json": str(ocr_candidates_path) if ocr_candidates_path else None,
        "scene_annotations_json": str(annotations_path) if annotation_records else None,
        "fused_scenes_json": str(scenes_dir / "fused_scenes.json"),
        "boundary_candidates_json": str(boundary_candidates_path),
        "cm_candidates_json": str(cm_candidates_path),
        "evidence_json": str(evidence_path),
        "metadata_json": str(metadata_path),
        "v03_style_metadata_json": str(v03_metadata_path),
        "frames_manifest_json": str(frames_manifest_path),
        "diagnostics_json": str(diagnostics_path),
        "frames_dir": str(frames_dir),
        "summary_json": str(summary_path),
        **export_files,
    }
    summary = {
        "metadata_path": str(metadata_path),
        "v03_style_metadata_path": str(v03_metadata_path),
        "captions_json": str(captions_path) if captions_path else None,
        "asr_json": str(asr_path) if asr_path else None,
        "frame_annotations_json": str(annotations_path) if annotation_records else None,
        "evidence_json": str(evidence_path),
        "frames_dir": str(frames_dir),
        "n_caption_segments": len(caption_segments),
        "n_asr_segments": len(asr_segments),
        "n_primary_segments": len(primary_segments),
        "text_merge_mode": args.text_merge_mode,
        "boundary_hints_mode": args.boundary_hints_mode,
        "finalizer_prompt": str(args.finalizer_prompt or FINALIZER_PROMPT_PATH),
        "scene_refine": bool(args.scene_refine),
        "n_boundary_hints": len(boundary_hints),
        "n_cues": len(cue_records),
        "n_scenes": len(scene_records),
        "n_boundary_candidates": len(boundary_candidates),
        "n_frame_annotations": len(annotation_records),
        "n_evidence_windows": len(evidence_windows),
        "n_topics": len(metadata.topics),
        "n_non_editorial_segments": len(metadata.non_editorial_segments),
        "warnings": metadata.warnings,
        "elapsed_sec": round(time.time() - started, 1),
        "artifacts": artifacts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    status(f"  wrote summary: {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production Japanese TV news TS metadata pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Input TS/MP4/WAV file")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--output", type=Path, default=None, help="Final ProgramMetadata JSON path")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--program-id", default=None)
    parser.add_argument("--channel-name", default=None)
    parser.add_argument("--program-name", default=None)
    parser.add_argument("--broadcast-start", default=None)
    parser.add_argument("--broadcast-end", default=None)
    parser.add_argument("--logical-start-sec", type=float, default=None,
                        help="Original-media start of a zero-based logical program window")
    parser.add_argument("--logical-end-sec", type=float, default=None,
                        help="Original-media end of a zero-based logical program window")
    parser.add_argument("--source-sha256", default=None,
                        help="Previously verified source hash; avoids re-reading a large source TS")

    parser.add_argument("--caption-mode", choices=["auto", "extract", "skip"], default="extract")
    parser.add_argument("--captions-json", type=Path, default=None)
    parser.add_argument("--caption-sid", type=int, default=None)
    parser.add_argument("--caption-accurate", action="store_true")
    parser.add_argument("--caption-lag-sec", type=float, default=4.5)
    parser.add_argument("--keep-ass", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Continue when required stages such as captions fail")
    parser.add_argument("--resume", action="store_true", help="Reuse existing stage artifacts from --out-dir when valid, including captions, ASR, OCR, scenes, frames, and VLM annotations")
    parser.add_argument("--skip-media-audio", action="store_true", help="Skip writing media/audio.wav; useful for fast smoke tests")

    parser.add_argument("--asr-mode", choices=["skip", "faster-whisper"], default="faster-whisper")
    parser.add_argument("--asr-json", type=Path, default=None)
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--asr-compute-type", default="float16")
    parser.add_argument("--keep-wav", action="store_true")

    parser.add_argument("--ocr-mode", choices=["auto", "rapidocr", "skip"], default="auto")
    parser.add_argument("--ocr-fps", type=float, default=1.0)
    parser.add_argument("--ocr-threshold", type=float, default=0.50)
    parser.add_argument("--ocr-merge-gap", type=int, default=2)
    parser.add_argument("--scene-mode", choices=["pyscenedetect", "skip"], default="pyscenedetect")
    parser.add_argument("--scene-strategy", choices=["pysd_only", "option_d", "ocr_only"], default="option_d")
    parser.add_argument("--scene-threshold", type=float, default=0.42)
    parser.add_argument("--scene-min-len-frames", type=int, default=90)
    parser.add_argument("--scene-refine", action="store_true",
                        help="Scene post-process: ContentDetector second pass inside scenes >20s, "
                             "snap new cuts to silence ends (±1s), merge <2s, force-split >30s")

    parser.add_argument("--vlm-mode", choices=["openai", "skip"], default="openai")
    parser.add_argument("--vlm-model", default="gpt-5.4-mini")
    parser.add_argument("--vlm-concurrency", type=int, default=1,
                        help="Concurrent visual-annotation requests (default: 1). "
                             "Frames are independent; increase within service rate limits.")
    parser.add_argument("--frame-interval-sec", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-frame-dimension", type=int, default=1280)
    parser.add_argument("--frame-annotations", type=Path, default=None)
    parser.add_argument("--reuse-annotations", action="store_true")

    parser.add_argument("--text-merge-mode", choices=["caption_priority", "ch140_fusion"], default="caption_priority", help="primary text timeline: caption_priority (default) or LLM fusion of ASR timing + caption lexicon")
    parser.add_argument("--fusion-model", default="gpt-5.5")
    parser.add_argument("--fusion-batch-target-tokens", type=int, default=2000)
    parser.add_argument("--fusion-caption-window-sec", type=float, default=45.0)
    parser.add_argument("--fusion-glossary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fusion-max-completion-tokens", type=int, default=8192)

    parser.add_argument("--boundary-hints-mode", choices=["none", "llm_text"], default="none", help="topic boundary hints: none (default) or LLM text segmentation over the primary timeline")
    parser.add_argument("--boundary-model", default="gpt-5.5")
    parser.add_argument("--boundary-chunk-target-tokens", type=int, default=6000)
    parser.add_argument("--boundary-chunk-overlap-sec", type=float, default=90.0)
    parser.add_argument("--boundary-merge-tolerance-sec", type=float, default=15.0)
    parser.add_argument("--boundary-snap-tolerance-sec", type=float, default=8.0)

    parser.add_argument("--evidence-window-sec", type=float, default=60.0)
    parser.add_argument("--finalizer", choices=["openai", "heuristic"], default="openai")
    parser.add_argument("--finalizer-model", default="gpt-5.5")
    parser.add_argument("--finalizer-prompt", type=Path, default=None,
                        help=f"Finalizer prompt file (default: {FINALIZER_PROMPT_PATH.name})")
    # Reasoning tokens count toward the completion budget, so leave enough
    # capacity for a one-hour program.
    parser.add_argument("--finalizer-max-completion-tokens", type=int, default=32000)
    # Keep enough evidence windows to cover the end of a one-hour program.
    parser.add_argument("--finalizer-max-windows", type=int, default=200)
    parser.add_argument("--allow-heuristic-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-topic-duration-sec", type=float, default=60.0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(message)s")
    summary = run_pipeline(args)
    status("\nDone")
    status("  summary_json:")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
