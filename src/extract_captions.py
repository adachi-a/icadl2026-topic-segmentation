"""Extract ARIB B24 captions from broadcast TS into CaptionSegment JSON.

This wrapper invokes ``assdumper/`` at the project root as a subprocess and
converts its ASS output into a list of Pydantic ``CaptionSegment`` models.

Pipeline:
    TS file ──assdumper──▶ ASS ──parse──▶ list[CaptionSegment] ──▶ JSON

CLI:
    python -m src.extract_captions \\
        --ts data/ts/<basename>.ts \\
        --out-dir data/captions/<basename>/ \\
        [--sid 1024] [--accurate] [--no-keep-ass]

When ``--sid`` is omitted, the service ID is detected from
``<ts>.program.txt`` encoded as cp932.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.topic_schema import CaptionSegment

LOG = logging.getLogger(__name__)

ASSDUMPER_DIR = Path(__file__).resolve().parent.parent / "assdumper"

# ASS Dialogue line: Dialogue: 0,0:00:00.78,0:00:05.45,nsz,,0,0,0,,<text-with-tags>
DIALOGUE_RE = re.compile(
    r"^Dialogue:\s*\d+,([\d:.]+),([\d:.]+),(\w+),[^,]*,\d+,\d+,\d+,[^,]*,(.*)$"
)
TAG_RE = re.compile(r"\{\\[^}]*\}")
POS_RE = re.compile(r"\\pos\((\d+),(\d+)\)")
# Detect a 1-12 character speaker name enclosed in full-width parentheses.
SPEAKER_RE = re.compile(r"^（([^）]{1,12})）\s*")


def parse_ass_time(s: str) -> float:
    """Convert ``0:12:34.56`` to 754.56 seconds."""
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def detect_sid(program_txt: Path) -> Optional[int]:
    """Extract ServiceID from a cp932-encoded ``<ts>.program.txt`` file."""
    raw = program_txt.read_bytes()
    text = raw.decode("cp932", errors="replace")
    m = re.search(r"ServiceID:(\d+)", text)
    return int(m.group(1)) if m else None


def run_assdumper(
    ts_path: Path,
    sid: int,
    ass_out: Path,
    accurate: bool = False,
) -> int:
    """Invoke assdumper as a subprocess and produce an ASS file.

    The subprocess uses ``cwd=ASSDUMPER_DIR`` so relative imports such as
    ``mpeg2ts.*`` resolve correctly. The return value is assdumper's exit code.
    """
    ass_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "assdumper.py",
        "-s", str(sid),
        "-i", str(ts_path.resolve()),
        "-o", str(ass_out.resolve()),
    ]
    if accurate:
        cmd.append("--accurate")
    LOG.info("running assdumper: sid=%d ts=%s", sid, ts_path.name)
    result = subprocess.run(
        cmd, cwd=ASSDUMPER_DIR, capture_output=True, text=True
    )
    if result.returncode != 0:
        LOG.warning("assdumper exited with code %d", result.returncode)
        if result.stderr:
            LOG.warning("stderr tail:\n%s", result.stderr[-1000:])
    if ass_out.exists():
        LOG.info("ass output: %d bytes", ass_out.stat().st_size)
    return result.returncode


def parse_ass_dialogues(ass_path: Path):
    """Yield ASS Dialogue rows as (start_sec, end_sec, style, raw_text, y_pos)."""
    with ass_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = DIALOGUE_RE.match(line.rstrip("\n"))
            if not m:
                continue
            start_str, end_str, style, raw = m.groups()
            pos_match = POS_RE.search(raw)
            y = int(pos_match.group(2)) if pos_match else 0
            yield parse_ass_time(start_str), parse_ass_time(end_str), style, raw, y


def strip_tags(text: str) -> str:
    """Remove all ASS override blocks (``{\\...}``) and surrounding whitespace."""
    return TAG_RE.sub("", text).strip()


def extract_speaker(text: str) -> tuple[Optional[str], str]:
    """Split a leading ``（NAME）`` speaker label from the caption text."""
    m = SPEAKER_RE.match(text)
    if not m:
        return None, text
    return m.group(1), text[m.end():].strip()


def parse_ass_to_segments(ass_path: Path) -> list[CaptionSegment]:
    """Convert an ASS file into a list of ``CaptionSegment`` records.

    Each Dialogue row becomes one segment. Empty tag-only placeholders are
    skipped. Multiple rows with identical start/end times remain separate to
    preserve split caption lines and multiple speakers.
    """
    segments: list[CaptionSegment] = []
    for start, end, _style, raw, _y in parse_ass_dialogues(ass_path):
        text = strip_tags(raw)
        if not text:
            continue
        speaker, body = extract_speaker(text)
        if not body:
            continue
        segments.append(
            CaptionSegment(
                start_sec=start,
                end_sec=end,
                text=body,
                speaker=speaker,
            )
        )
    return segments


def extract_captions(
    ts_path: Path,
    out_dir: Path,
    sid: Optional[int] = None,
    accurate: bool = False,
    keep_ass: bool = True,
) -> Path:
    """End-to-end: TS → ASS → CaptionSegment JSON。

    Returns:
        Path to the generated JSON file.
    """
    if sid is None:
        program_txt = ts_path.with_suffix(ts_path.suffix + ".program.txt")
        if program_txt.exists():
            sid = detect_sid(program_txt)
            if sid is not None:
                LOG.info("auto-detected SID=%d from %s", sid, program_txt.name)
        if sid is None:
            raise ValueError(
                f"SID not provided and could not auto-detect from "
                f"{ts_path.name}.program.txt. Pass --sid explicitly."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    base = ts_path.stem
    ass_path = out_dir / f"{base}.ass"
    json_path = out_dir / f"{base}.captions.json"

    rc = run_assdumper(ts_path, sid, ass_path, accurate=accurate)
    if not ass_path.exists() or ass_path.stat().st_size == 0:
        raise RuntimeError(
            f"assdumper produced no output (exit={rc}). See stderr above."
        )

    segments = parse_ass_to_segments(ass_path)
    payload = {
        "ts_filename": ts_path.name,
        "service_id": sid,
        "assdumper_exit_code": rc,
        "n_segments": len(segments),
        "segments": [s.model_dump() for s in segments],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not keep_ass:
        ass_path.unlink()

    LOG.info("wrote %d segments → %s", len(segments), json_path)
    return json_path


def main():
    ap = argparse.ArgumentParser(
        description="ARIB B24 caption extractor (TS → CaptionSegment JSON)"
    )
    ap.add_argument("--ts", type=Path, required=True, help="Input TS file")
    ap.add_argument(
        "--sid",
        type=int,
        help="Service ID (auto-detected from <ts>.program.txt if omitted)",
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    ap.add_argument(
        "--accurate", action="store_true", help="Pass --accurate to assdumper"
    )
    ap.add_argument(
        "--no-keep-ass",
        action="store_true",
        help="Delete intermediate .ass after parsing",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    extract_captions(
        ts_path=args.ts,
        out_dir=args.out_dir,
        sid=args.sid,
        accurate=args.accurate,
        keep_ass=not args.no_keep_ass,
    )


if __name__ == "__main__":
    main()
