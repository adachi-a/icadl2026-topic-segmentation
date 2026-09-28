"""Fuse ASR timing with caption-derived lexical evidence.

ASR timestamps are preserved while names, places, and homophones may be
corrected using captions. Long programs are processed in time-windowed batches,
with resumable batch outputs and fallback to the original ASR text."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from src.topic_schema import ASRSegment, CaptionSegment, UnifiedTranscript

LOG = logging.getLogger(__name__)

# Default model and prompt.
DEFAULT_REFINEMENT_MODEL = "gpt-5.5"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "transcript_unified_stage1.txt"
PROMPT_FILE_WINDOWED = (
    Path(__file__).resolve().parent.parent
    / "prompts" / "transcript_unified_stage1_windowed.txt"
)

# Symmetric margin around each batch for selecting caption evidence.
DEFAULT_CAPTION_WINDOW_SEC = 45.0

# Tiktoken encoding for current OpenAI models.
DEFAULT_ENCODING = "o200k_base"

# Token budget for the ASR portion of one input batch. Caption evidence and
# the system prompt are cached separately by the API.
DEFAULT_BATCH_TARGET_TOKENS = 2000
# Completion-token budget, with headroom for output close to the input size.
DEFAULT_MAX_COMPLETION_TOKENS = 8192

# Caption-token limit and term limit for one glossary extraction call.
DEFAULT_GLOSSARY_INPUT_LIMIT_TOKENS = 60000
DEFAULT_GLOSSARY_MAX_TERMS = 300


# ============================================================
# Internal Pydantic models for Structured Outputs.
# ============================================================

class _LLMItemIn(BaseModel):
    """One model input segment containing only its ID and text."""
    id: int
    text: str


class _LLMBatchOut(BaseModel):
    """Model output containing IDs and corrected text."""
    items: list[_LLMItemIn]


class _GlossaryOut(BaseModel):
    """Structured glossary-extraction output."""
    terms: list[str]


# ============================================================
# I/O helpers
# ============================================================

def _load_asr(asr_json: Path) -> tuple[list[dict], str, float]:
    """Load ASR JSON as ``(segments, model_name, duration_sec)``."""
    data = json.loads(asr_json.read_text(encoding="utf-8"))
    if isinstance(data, list):  # List-only input format.
        return data, "large-v3", round(data[-1]["end"], 2) if data else 0.0
    return data["segments"], data.get("model", "large-v3"), data.get("duration_sec", 0.0)


@dataclass(frozen=True)
class CaptionEntry:
    """One timed caption line used for windowed lexical evidence."""
    start: float
    end: float
    text: str


def _load_caption_entries(captions_json: Path) -> list[CaptionEntry]:
    """Load timed captions, normalize with NFKC, and discard empty lines."""
    data = json.loads(captions_json.read_text(encoding="utf-8"))
    raw_segments = data.get("segments", data if isinstance(data, list) else [])
    entries: list[CaptionEntry] = []
    for item in raw_segments:
        text = unicodedata.normalize("NFKC", str(item.get("text", ""))).strip()
        if not text:
            continue
        start = float(item.get("start_sec", item.get("start", 0.0)))
        end = float(item.get("end_sec", item.get("end", start)))
        entries.append(CaptionEntry(start=start, end=end, text=text))
    return entries


def _captions_in_window(
    entries: list[CaptionEntry],
    start: float,
    end: float,
    margin_sec: float,
) -> str:
    """Join caption lines overlapping ``[start-margin, end+margin]``."""
    lo = start - margin_sec
    hi = end + margin_sec
    return "\n".join(e.text for e in entries if e.end >= lo and e.start <= hi)


_GLOSSARY_SYSTEM_PROMPT = (
    "あなたは報道番組アーカイブの索引編集者です。与えられたテレビ番組の字幕テキストから、"
    "人名・地名・組織名・商品名・番組名・専門用語などの固有名詞を、"
    "字幕に現れた正確な表記のまま重複なく抽出してください。"
    "一般名詞・動詞・数値のみの表現は含めないでください。"
    "出力は指定された JSON 構造のみとします。"
)


def _split_text_by_tokens(text: str, encoding_name: str, limit_tokens: int) -> list[str]:
    """Split text by line into chunks that fit the token limit."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    if len(enc.encode(text)) <= limit_tokens:
        return [text]
    chunks: list[str] = []
    cur_lines: list[str] = []
    cur_toks = 0
    for line in text.split("\n"):
        n = len(enc.encode(line + "\n"))
        if cur_lines and cur_toks + n > limit_tokens:
            chunks.append("\n".join(cur_lines))
            cur_lines, cur_toks = [], 0
        cur_lines.append(line)
        cur_toks += n
    if cur_lines:
        chunks.append("\n".join(cur_lines))
    return chunks


def _merge_glossaries(term_lists: list[list[str]], max_terms: int) -> list[str]:
    """Deduplicate chunk glossaries in order and truncate to ``max_terms``."""
    seen: set[str] = set()
    merged: list[str] = []
    for terms in term_lists:
        for term in terms:
            key = unicodedata.normalize("NFKC", term).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(key)
            if len(merged) >= max_terms:
                return merged
    return merged


def _glossary_llm_call(chunk_text: str, model: str, max_terms: int) -> list[str]:
    """Extract proper nouns from one caption chunk."""
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    response = client.beta.chat.completions.parse(
        model=model,
        max_completion_tokens=4096,
        messages=[
            {"role": "system", "content": _GLOSSARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"最大 {max_terms} 件。\n\n## 字幕テキスト\n{chunk_text}"},
        ],
        response_format=_GlossaryOut,
    )
    msg = response.choices[0].message
    if msg.refusal or msg.parsed is None:
        LOG.warning("glossary extraction returned no result (refusal=%s)", msg.refusal)
        return []
    return msg.parsed.terms


def _extract_glossary(
    caption_text: str,
    model: str,
    encoding_name: str,
    limit_tokens: int = DEFAULT_GLOSSARY_INPUT_LIMIT_TOKENS,
    max_terms: int = DEFAULT_GLOSSARY_MAX_TERMS,
) -> list[str]:
    """Extract a program glossary; treat failed chunks as empty."""
    chunks = _split_text_by_tokens(caption_text, encoding_name, limit_tokens)
    term_lists: list[list[str]] = []
    for ci, chunk in enumerate(chunks):
        try:
            terms = _glossary_llm_call(chunk, model, max_terms)
        except Exception as e:
            LOG.warning("glossary chunk %d failed: %s", ci, e)
            terms = []
        term_lists.append(terms)
    return _merge_glossaries(term_lists, max_terms)


def _load_prompt() -> str:
    return PROMPT_FILE_WINDOWED.read_text(encoding="utf-8").strip()


# ============================================================
# Tiktoken-driven batching
# ============================================================

def _make_batches(
    items: list[_LLMItemIn],
    encoding_name: str,
    target_tokens: int,
) -> list[list[_LLMItemIn]]:
    """Pack ASR segments into dynamic token-limited batches.

    Each batch stays approximately below ``target_tokens``. A segment that
    exceeds the target forms a batch by itself.
    """
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    batches: list[list[_LLMItemIn]] = []
    cur: list[_LLMItemIn] = []
    cur_toks = 0
    for item in items:
        # Include both ID and text to approximate the serialized JSON cost.
        n = len(enc.encode(json.dumps(item.model_dump(), ensure_ascii=False)))
        if cur and cur_toks + n > target_tokens:
            batches.append(cur)
            cur, cur_toks = [], 0
        cur.append(item)
        cur_toks += n
    if cur:
        batches.append(cur)
    return batches


def _batch_time_range(
    batch: list[_LLMItemIn],
    id_to_orig: dict[int, dict],
) -> tuple[float, float]:
    """Return the minimum start and maximum end within an ASR batch."""
    starts = [float(id_to_orig[it.id]["start"]) for it in batch if it.id in id_to_orig]
    ends = [float(id_to_orig[it.id]["end"]) for it in batch if it.id in id_to_orig]
    if not starts:
        return (0.0, 0.0)
    return (min(starts), max(ends))


def _build_user_content(glossary: list[str], window_text: str, input_json: str) -> str:
    """Build a batch user message, omitting an empty glossary section."""
    parts: list[str] = []
    if glossary:
        parts.append(
            "## 番組グロッサリ (番組全体の固有名詞、正しい表記の一覧)\n"
            + "、".join(glossary)
        )
    parts.append(
        "## 参考資料: この時間帯の字幕データ (lexical 辞書、時間情報は無視)\n"
        + window_text
    )
    parts.append(
        "## 修正対象データ (id + text の JSON、id を保ったまま text のみ修正)\n"
        + input_json
    )
    return "\n\n".join(parts)


# ============================================================
# Model invocation
# ============================================================

def _refine_batch(
    batch: list[_LLMItemIn],
    system_prompt: str,
    dictionary_text: str,
    model: str,
    max_completion_tokens: int,
) -> tuple[Optional[list[_LLMItemIn]], dict]:
    """Correct one batch and return None when the caller should fall back.

    ``dictionary_text`` is the complete user message built by
    ``_build_user_content``.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    response = client.beta.chat.completions.parse(
        model=model,
        max_completion_tokens=max_completion_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": dictionary_text},
        ],
        response_format=_LLMBatchOut,
    )
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    msg = response.choices[0].message
    if msg.refusal:
        LOG.warning("LLM refused batch: %s", msg.refusal)
        return None, usage
    if msg.parsed is None:
        LOG.warning("LLM returned empty result")
        return None, usage
    return msg.parsed.items, usage


# ============================================================
# Pipeline
# ============================================================

def refine_transcript_segments(
    asr_segments: list[dict],
    caption_entries: list[CaptionEntry],
    *,
    out_path: Path,
    program_id: str,
    asr_model: str = "large-v3",
    duration_sec: float = 0.0,
    backup_dir: Optional[Path] = None,
    model: str = DEFAULT_REFINEMENT_MODEL,
    encoding_name: str = DEFAULT_ENCODING,
    batch_target_tokens: int = DEFAULT_BATCH_TARGET_TOKENS,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    caption_window_sec: float = DEFAULT_CAPTION_WINDOW_SEC,
    use_glossary: bool = True,
    glossary_path: Optional[Path] = None,
) -> UnifiedTranscript:
    """Correct and save ASR segments using windowed captions and a glossary.

    asr_segments: [{"id": int, "start": float, "end": float, "text": str}, ...]
    caption_entries: Caption lines with lag-adjusted timestamps.
    """
    system_prompt = _load_prompt()

    if duration_sec <= 0.0 and asr_segments:
        duration_sec = round(float(asr_segments[-1]["end"]), 2)

    LOG.info(
        "loaded ASR: %d segments (duration %.1fs, model %s)",
        len(asr_segments), duration_sec, asr_model,
    )
    LOG.info("loaded captions: %d entries (time-windowed lexical dictionary)",
             len(caption_entries))

    # Reuse a saved program glossary when available.
    glossary: list[str] = []
    if use_glossary and caption_entries:
        if glossary_path is not None and glossary_path.exists():
            glossary = json.loads(glossary_path.read_text(encoding="utf-8"))["terms"]
            LOG.info("reusing glossary: %d terms from %s", len(glossary), glossary_path)
        else:
            caption_text = "\n".join(e.text for e in caption_entries)
            glossary = _extract_glossary(
                caption_text, model=model, encoding_name=encoding_name,
            )
            LOG.info("extracted glossary: %d terms", len(glossary))
            if glossary_path is not None:
                glossary_path.parent.mkdir(parents=True, exist_ok=True)
                glossary_path.write_text(
                    json.dumps({"terms": glossary}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    items = [
        _LLMItemIn(id=int(s["id"]), text=unicodedata.normalize("NFKC", s["text"]))
        for s in asr_segments
    ]
    id_to_orig = {int(s["id"]): s for s in asr_segments}

    batches = _make_batches(items, encoding_name=encoding_name, target_tokens=batch_target_tokens)
    LOG.info("batched %d segments into %d batches (target=%d tokens)",
             len(items), len(batches), batch_target_tokens)

    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)

    refined_items: list[_LLMItemIn] = []
    n_fallback = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    t0 = time.time()

    for bi, batch in enumerate(batches):
        backup_path = (
            backup_dir / f"batch_{bi:04d}.json" if backup_dir is not None else None
        )
        if backup_path is not None and backup_path.exists():
            try:
                cached = json.loads(backup_path.read_text(encoding="utf-8"))
                cached_items = [_LLMItemIn(**x) for x in cached.get("items", [])]
                if len(cached_items) == len(batch):
                    refined_items.extend(cached_items)
                    LOG.info("[%d/%d] using cached backup (%d items)",
                             bi + 1, len(batches), len(cached_items))
                    continue
            except Exception as e:
                LOG.warning("backup read failed (%s); re-running batch", e)

        b_start, b_end = _batch_time_range(batch, id_to_orig)
        window_text = _captions_in_window(
            caption_entries, b_start, b_end, margin_sec=caption_window_sec,
        )
        input_json = json.dumps([s.model_dump() for s in batch], ensure_ascii=False)
        dictionary_text = _build_user_content(glossary, window_text, input_json)

        LOG.info("[%d/%d] refining batch (%d items, %.1f-%.1fs)",
                 bi + 1, len(batches), len(batch), b_start, b_end)
        try:
            corrected, usage = _refine_batch(
                batch=batch,
                system_prompt=system_prompt,
                dictionary_text=dictionary_text,
                model=model,
                max_completion_tokens=max_completion_tokens,
            )
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
        except Exception as e:
            LOG.warning("batch %d failed: %s — falling back to raw ASR", bi, e)
            corrected = None

        if corrected is None or len(corrected) != len(batch):
            if corrected is not None:
                LOG.warning(
                    "batch %d: LLM returned %d items (expected %d); using raw ASR",
                    bi, len(corrected), len(batch),
                )
            n_fallback += 1
            corrected = batch  # Preserve the original text on fallback.

        refined_items.extend(corrected)
        if backup_path is not None:
            backup_path.write_text(
                json.dumps(
                    {"items": [c.model_dump() for c in corrected]},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )

    elapsed = time.time() - t0
    LOG.info(
        "stage 1 done: %d batches (%d fallback), %.1fs, tokens=%d",
        len(batches), n_fallback, elapsed, total_usage.get("total_tokens", 0),
    )

    # Restore ASR start/end times and assemble the output segments.
    out_segments: list[ASRSegment] = []
    for it in refined_items:
        orig = id_to_orig.get(it.id)
        if orig is None:
            LOG.warning("id %d not found in original ASR; skipping", it.id)
            continue
        out_segments.append(ASRSegment(
            start=float(orig["start"]),
            end=float(orig["end"]),
            text=it.text,
        ))

    result = UnifiedTranscript(
        program_id=program_id,
        asr_model=asr_model,
        refinement_model=model,
        duration_sec=duration_sec,
        n_input_segments=len(asr_segments),
        n_output_segments=len(out_segments),
        n_batches=len(batches),
        n_fallback_batches=n_fallback,
        segments=out_segments,
        metadata={
            "encoding": encoding_name,
            "batch_target_tokens": batch_target_tokens,
            "caption_window_sec": caption_window_sec,
            "n_caption_entries": len(caption_entries),
            "n_glossary_terms": len(glossary),
            "elapsed_sec": round(elapsed, 1),
            "token_usage": total_usage,
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOG.info("wrote %s", out_path)
    return result


def refine_transcript(
    asr_json: Path,
    captions_json: Path,
    out_path: Path,
    backup_dir: Optional[Path] = None,
    program_id: Optional[str] = None,
    model: str = DEFAULT_REFINEMENT_MODEL,
    encoding_name: str = DEFAULT_ENCODING,
    batch_target_tokens: int = DEFAULT_BATCH_TARGET_TOKENS,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    caption_window_sec: float = DEFAULT_CAPTION_WINDOW_SEC,
    use_glossary: bool = True,
) -> UnifiedTranscript:
    """Thin file-based wrapper used by the standalone CLI."""
    load_dotenv()
    raw_segments, asr_model, duration = _load_asr(asr_json)
    for i, s in enumerate(raw_segments):
        s.setdefault("id", i)
    caption_entries = _load_caption_entries(captions_json)
    pid = program_id or asr_json.stem.replace(".asr", "")
    glossary_path = backup_dir / "glossary.json" if backup_dir is not None else None
    return refine_transcript_segments(
        raw_segments,
        caption_entries,
        out_path=out_path,
        program_id=pid,
        asr_model=asr_model,
        duration_sec=duration,
        backup_dir=backup_dir,
        model=model,
        encoding_name=encoding_name,
        batch_target_tokens=batch_target_tokens,
        max_completion_tokens=max_completion_tokens,
        caption_window_sec=caption_window_sec,
        use_glossary=use_glossary,
        glossary_path=glossary_path,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Refine ASR text with caption lexical context"
    )
    ap.add_argument("--asr", type=Path, required=True, help="ASR JSON (.asr.json)")
    ap.add_argument("--captions", type=Path, required=True, help="Captions JSON (.captions.json)")
    ap.add_argument("--out", type=Path, required=True, help="Output transcript_unified.json")
    ap.add_argument("--backup-dir", type=Path, default=None, help="Per-batch backup dir (resume)")
    ap.add_argument("--program-id", type=str, default=None, help="program_id (default: ASR file stem)")
    ap.add_argument("--model", default=DEFAULT_REFINEMENT_MODEL, help="Azure OpenAI deployment name")
    ap.add_argument("--encoding", default=DEFAULT_ENCODING, help="tiktoken encoding")
    ap.add_argument("--batch-target-tokens", type=int, default=DEFAULT_BATCH_TARGET_TOKENS)
    ap.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    ap.add_argument("--caption-window-sec", type=float, default=DEFAULT_CAPTION_WINDOW_SEC,
                    help="Caption-evidence margin around each batch in seconds")
    ap.add_argument("--glossary", action=argparse.BooleanOptionalAction, default=True,
                    help="Extract a program glossary with the language model")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    refine_transcript(
        asr_json=args.asr,
        captions_json=args.captions,
        out_path=args.out,
        backup_dir=args.backup_dir,
        program_id=args.program_id,
        model=args.model,
        encoding_name=args.encoding,
        batch_target_tokens=args.batch_target_tokens,
        max_completion_tokens=args.max_completion_tokens,
        caption_window_sec=args.caption_window_sec,
        use_glossary=args.glossary,
    )


if __name__ == "__main__":
    main()
