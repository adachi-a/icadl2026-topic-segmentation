"""Data models shared by captions, ASR, and the fused transcript."""

from typing import Optional

from pydantic import BaseModel, Field


class ASRSegment(BaseModel):
    start: float
    end: float
    text: str


class CaptionSegment(BaseModel):
    """One ARIB B24 caption segment."""

    start_sec: float
    end_sec: float
    text: str
    speaker: Optional[str] = None


class UnifiedTranscript(BaseModel):
    """Transcript that retains ASR timing and uses captions as lexical evidence."""

    program_id: str
    asr_model: str
    refinement_model: str
    duration_sec: float
    n_input_segments: int
    n_output_segments: int
    n_batches: int
    n_fallback_batches: int
    segments: list[ASRSegment] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
