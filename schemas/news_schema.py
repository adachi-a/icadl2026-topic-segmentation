"""Pydantic models for public program metadata and visual annotations.

All public fields are defined here. Extra fields are rejected, and schemas for
Structured Outputs require every field with `additionalProperties: false`.
Visual annotations contain visual evidence; `ProgramMetadata` contains the
final program-level output."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Domain = Literal[
    "arts_culture_entertainment_media",
    "conflict_war_peace",
    "crime_law_justice",
    "disaster_accident_emergency",
    "economy_business_finance",
    "education",
    "environment",
    "health",
    "human_interest",
    "labour",
    "lifestyle_leisure",
    "politics",
    "religion",
    "science_technology",
    "society",
    "sport",
    "weather",
]

PresentationForm = Literal[
    "anchor_or_studio_read",
    "news_package",
    "reporter_report",
    "interview_or_comment",
    "press_conference_or_statement",
    "studio_explanation_or_analysis",
    "event_or_field_footage",
    "document_or_graphics",
    "weather_forecast",
    "sports_results_or_highlights",
    "market_or_financial_data",
    "product_or_service_feature",
    "other_editorial",
    "none_non_editorial",
]

ShotType = Literal[
    "studio",
    "field_or_event",
    "reporter_or_interview",
    "press_conference_or_official",
    "document_graphics_or_map",
    "weather",
    "sports",
    "market_board",
    "product_demo",
    "non_editorial",
    "other",
]

SponsorPresentation = Literal[
    "none",
    "overlay_during_editorial",
    "fullscreen_sponsor_credit",
    "ambiguous",
]

NonEditorialKind = Literal[
    "none",
    "commercial",
    "program_promo",
    "sponsor",
    "opening_or_ending",
    "other_non_editorial",
    "ambiguous",
]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrameVisualAnnotation(StrictBaseModel):
    """VLM output for one representative frame.

    All fields are required for OpenAI Structured Outputs compatibility.
    Nullable fields must be explicitly returned as null when unknown.
    """

    frame_id: str = Field(description="Stable ID of the representative frame.")
    scene_id: str | None = Field(description="Scene ID, or null when the frame is not yet assigned to a scene.")
    timestamp_sec: float | None = Field(description="Timestamp in seconds from the program start, or null if unavailable.")
    frame_index: int | None = Field(description="Video frame index, or null if unavailable.")

    ocr_texts: list[str] = Field(description="Visible Japanese text strings read from the frame. Use an empty list if none is readable.")
    primary_onscreen_text: str | None = Field(description="Most important headline/telop text, or null if there is no clear primary text.")
    visual_summary: str = Field(description="One-sentence description of what is visible in the frame.")
    people_or_speakers_visible: str | None = Field(description="Brief description of visible people/speakers and visible roles. Do not identify private people unless text on screen identifies them.")

    shot_type: ShotType = Field(description="Coarse visual form of the frame. This is evidence, not a final content decision.")
    sponsor_presentation: SponsorPresentation = Field(description="Whether the frame contains sponsor presentation or credit display.")
    non_editorial_kind: NonEditorialKind = Field(description="Non-editorial type if the frame appears to be CM/promo/sponsor/etc.; otherwise none.")
    is_probably_editorial: bool = Field(description="True if the frame appears to be part of news editorial content.")

    likely_presentation_form: PresentationForm = Field(description="Best visual-only hint of the presentation form, or none_non_editorial for non-editorial frames.")
    domain_hints: list[Domain] = Field(description="Up to three domain hints from visible information only. Use an empty list if unclear.", max_length=3)

    evidence: list[str] = Field(description="Short reasons for the chosen labels.")
    warnings: list[str] = Field(description="Uncertainties such as unreadable text or insufficient visual context.")


class Topic(StrictBaseModel):
    topic_id: str
    start_sec: float
    end_sec: float
    editorial: bool
    domain: Domain
    domain_alternatives: list[Domain] = Field(max_length=3)
    title: str
    summary: str
    interrupted_by_non_editorial_segment_ids: list[str]
    evidence_texts: list[str]
    notes: str | None


class NonEditorialSegment(StrictBaseModel):
    segment_id: str
    start_sec: float
    end_sec: float
    non_editorial_kind: NonEditorialKind
    summary: str | None
    source_scene_ids: list[str]
    representative_frame_ids: list[str]
    notes: str | None


class ProgramMetadata(StrictBaseModel):
    program_id: str
    source_ts_path: str
    channel_name: str | None
    program_name: str | None
    broadcast_start: str | None
    broadcast_end: str | None
    topics: list[Topic]
    non_editorial_segments: list[NonEditorialSegment]
    warnings: list[str]


def openai_response_format_for_frame_annotation() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "frame_visual_annotation",
            "strict": True,
            "schema": FrameVisualAnnotation.model_json_schema(),
        },
    }


def openai_response_format_for_program_metadata() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "program_metadata",
            "strict": True,
            "schema": ProgramMetadata.model_json_schema(),
        },
    }


def program_metadata_json_schema() -> dict:
    return ProgramMetadata.model_json_schema()
