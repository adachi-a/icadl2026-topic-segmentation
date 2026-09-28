"""Detect Topic-boundary hints from the primary text timeline.

Overlapping timestamped chunks are analyzed independently, stitched, filtered,
and optionally snapped to visual candidates. The finalizer remains responsible
for the final Topic boundaries."""

from __future__ import annotations

import json
import logging
import os
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

LOG = logging.getLogger(__name__)

DEFAULT_BOUNDARY_MODEL = "gpt-5.5"
DEFAULT_ENCODING = "o200k_base"
# About 6,000 tokens per chunk yields four to five chunks for a one-hour
# program. A 90-second overlap exposes cross-chunk topics from both sides.
DEFAULT_CHUNK_TARGET_TOKENS = 6000
DEFAULT_CHUNK_OVERLAP_SEC = 90.0
# Merge tolerance for duplicate boundaries detected in an overlap.
DEFAULT_MERGE_TOLERANCE_SEC = 15.0
# Snap tolerance for visual scene/OCR boundary candidates.
DEFAULT_SNAP_TOLERANCE_SEC = 8.0
DEFAULT_MAX_COMPLETION_TOKENS = 4096

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "topic_boundary_ja_v2.txt"

# Confidence is an ordinal three-level scale. Continuous self-assessment by
# the model is poorly calibrated, so explicit transitions map to high,
# strong cues to medium, and content-only inference to low.
CONFIDENCE_LEVELS = ("low", "medium", "high")
_CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}


def _confidence_rank(confidence: str) -> int:
    return _CONFIDENCE_RANK.get(confidence, 0)


@dataclass(frozen=True)
class Chunk:
    """One chunk of formatted, timestamped text."""
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class BoundaryHint:
    """One detected topic-boundary hint with low/medium/high confidence."""
    time_sec: float
    title: str
    confidence: str
    raw_time_sec: float
    snapped_to: Optional[str] = None
    source: str = "llm_text"


def format_timeline_line(segment: dict) -> str:
    """Format a segment as ``[m:ss] text`` using minutes from program start."""
    total = int(segment["start"])
    text = unicodedata.normalize("NFKC", str(segment["text"])).strip()
    return f"[{total // 60}:{total % 60:02d}] {text}"


def make_overlapping_chunks(
    segments: list[dict],
    encoding_name: str,
    target_tokens: int,
    overlap_sec: float,
) -> list[Chunk]:
    """Split segments into token-limited chunks with temporal overlap.

    Each next chunk starts at the first segment within ``overlap_sec`` of the
    previous chunk's end. It always advances by at least one segment.
    """
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    chunks: list[Chunk] = []
    i = 0
    n = len(segments)
    while i < n:
        lines: list[str] = []
        toks = 0
        j = i
        while j < n:
            line = format_timeline_line(segments[j])
            t = len(enc.encode(line + "\n"))
            if lines and toks + t > target_tokens:
                break
            lines.append(line)
            toks += t
            j += 1
        chunk_end = float(segments[j - 1]["end"])
        chunks.append(Chunk(float(segments[i]["start"]), chunk_end, "\n".join(lines)))
        if j >= n:
            break
        k = j
        while k > i + 1 and float(segments[k - 1]["start"]) > chunk_end - overlap_sec:
            k -= 1
        i = k
    return chunks


def stitch_boundaries(
    per_chunk: list[list[BoundaryHint]],
    merge_tolerance_sec: float,
) -> list[BoundaryHint]:
    """Merge chunk results chronologically, keeping the strongest nearby hint."""
    all_hints = sorted((h for hints in per_chunk for h in hints), key=lambda h: h.time_sec)
    merged: list[BoundaryHint] = []
    for hint in all_hints:
        if merged and hint.time_sec - merged[-1].time_sec <= merge_tolerance_sec:
            if _confidence_rank(hint.confidence) > _confidence_rank(merged[-1].confidence):
                merged[-1] = hint
            continue
        merged.append(hint)
    return merged


def snap_to_candidates(
    hints: list[BoundaryHint],
    candidates: list[tuple[float, str]],
    snap_tolerance_sec: float,
) -> list[BoundaryHint]:
    """Snap to a visual candidate within tolerance while retaining the raw time."""
    snapped: list[BoundaryHint] = []
    for hint in hints:
        best: Optional[tuple[float, str]] = None
        for cand_time, cand_id in candidates:
            dist = abs(cand_time - hint.time_sec)
            if dist <= snap_tolerance_sec and (best is None or dist < abs(best[0] - hint.time_sec)):
                best = (cand_time, cand_id)
        if best is None:
            snapped.append(hint)
        else:
            snapped.append(BoundaryHint(
                time_sec=best[0], title=hint.title, confidence=hint.confidence,
                raw_time_sec=hint.raw_time_sec, snapped_to=best[1], source=hint.source,
            ))
    return snapped


def filter_hints(
    hints: list[BoundaryHint],
    non_editorial: list[tuple[float, float]],
    min_gap_sec: float,
) -> list[BoundaryHint]:
    """Remove boundaries inside non-editorial intervals and enforce spacing."""
    editorial = [
        h for h in hints
        if not any(start <= h.time_sec < end for start, end in non_editorial)
    ]
    kept: list[BoundaryHint] = []
    for hint in sorted(editorial, key=lambda h: h.time_sec):
        if kept and hint.time_sec - kept[-1].time_sec < min_gap_sec:
            if _confidence_rank(hint.confidence) > _confidence_rank(kept[-1].confidence):
                kept[-1] = hint
            continue
        kept.append(hint)
    return kept


# ============================================================
# Model invocation and public API
# ============================================================

class _BoundaryItem(BaseModel):
    """One boundary returned by the model."""
    time_sec: float
    title: str
    confidence: Literal["low", "medium", "high"]


class _BoundaryOut(BaseModel):
    """Structured model response."""
    boundaries: list[_BoundaryItem]


def _load_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


def _boundary_llm_call(chunk: Chunk, model: str, max_completion_tokens: int) -> tuple[list[BoundaryHint], dict]:
    """Analyze one chunk; refusals and empty responses return ``([], usage)``."""
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    user_content = (
        f"## 対象チャンク (番組内 {chunk.start_sec:.0f}s - {chunk.end_sec:.0f}s)\n"
        f"{chunk.text}"
    )
    response = client.beta.chat.completions.parse(
        model=model,
        max_completion_tokens=max_completion_tokens,
        messages=[
            {"role": "system", "content": _load_prompt()},
            {"role": "user", "content": user_content},
        ],
        response_format=_BoundaryOut,
    )
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    msg = response.choices[0].message
    if msg.refusal or msg.parsed is None:
        LOG.warning("boundary LLM returned no result (refusal=%s)", msg.refusal)
        return [], usage
    hints = [
        BoundaryHint(time_sec=float(b.time_sec), title=b.title.strip(),
                     confidence=b.confidence, raw_time_sec=float(b.time_sec))
        for b in msg.parsed.boundaries
    ]
    return hints, usage


def detect_topic_boundaries(
    segments: list[dict],
    *,
    out_path: Path,
    backup_dir: Optional[Path] = None,
    model: str = DEFAULT_BOUNDARY_MODEL,
    encoding_name: str = DEFAULT_ENCODING,
    chunk_target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    chunk_overlap_sec: float = DEFAULT_CHUNK_OVERLAP_SEC,
    merge_tolerance_sec: float = DEFAULT_MERGE_TOLERANCE_SEC,
    snap_tolerance_sec: float = DEFAULT_SNAP_TOLERANCE_SEC,
    min_gap_sec: float = 60.0,
    candidates: Optional[list[tuple[float, str]]] = None,
    non_editorial: Optional[list[tuple[float, float]]] = None,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> list[BoundaryHint]:
    """Detect topic-boundary hints from primary segments and save them.

    A failed chunk is skipped. When ``backup_dir`` is provided, per-chunk
    results are saved and reused on subsequent runs.
    """
    chunks = make_overlapping_chunks(
        segments, encoding_name=encoding_name,
        target_tokens=chunk_target_tokens, overlap_sec=chunk_overlap_sec,
    )
    LOG.info("boundary detection: %d segments -> %d chunks", len(segments), len(chunks))
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)

    per_chunk: list[list[BoundaryHint]] = []
    n_failed = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        backup_path = backup_dir / f"chunk_{ci:04d}.json" if backup_dir is not None else None
        if backup_path is not None and backup_path.exists():
            try:
                cached = json.loads(backup_path.read_text(encoding="utf-8"))
                per_chunk.append([BoundaryHint(**b) for b in cached["boundaries"]])
                LOG.info("[%d/%d] using cached chunk backup", ci + 1, len(chunks))
                continue
            except Exception as e:
                LOG.warning("chunk backup read failed (%s); re-running", e)
        try:
            hints, usage = _boundary_llm_call(chunk, model, max_completion_tokens)
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
        except Exception as e:
            LOG.warning("chunk %d failed: %s — skipping", ci, e)
            n_failed += 1
            per_chunk.append([])
            continue
        # Discard timestamps outside the current chunk.
        hints = [h for h in hints if chunk.start_sec <= h.time_sec <= chunk.end_sec]
        LOG.info("[%d/%d] chunk %.0f-%.0fs: %d boundaries",
                 ci + 1, len(chunks), chunk.start_sec, chunk.end_sec, len(hints))
        per_chunk.append(hints)
        if backup_path is not None:
            backup_path.write_text(
                json.dumps({"boundaries": [asdict(h) for h in hints]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    merged = stitch_boundaries(per_chunk, merge_tolerance_sec)
    snapped = snap_to_candidates(merged, candidates or [], snap_tolerance_sec)
    final = filter_hints(snapped, non_editorial or [], min_gap_sec)
    LOG.info("boundaries: %d raw -> %d stitched -> %d final (%d chunks failed)",
             sum(len(h) for h in per_chunk), len(merged), len(final), n_failed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "schema": "topic_boundary_hints_v1",
        "model": model,
        "n_chunks": len(chunks),
        "n_failed_chunks": n_failed,
        "n_boundaries": len(final),
        "params": {
            "chunk_target_tokens": chunk_target_tokens,
            "chunk_overlap_sec": chunk_overlap_sec,
            "merge_tolerance_sec": merge_tolerance_sec,
            "snap_tolerance_sec": snap_tolerance_sec,
            "min_gap_sec": min_gap_sec,
        },
        "token_usage": total_usage,
        "elapsed_sec": round(time.time() - t0, 1),
        "boundaries": [asdict(h) for h in final],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("wrote %s", out_path)
    return final
