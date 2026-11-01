"""Pydantic contracts: what the analyser must return per batch, what the database stores, what the API accepts.

Each reply is validated in two layers: the envelope (`BatchReply`) must parse, then every event is
validated on its own (`ReplyEvent`) so one malformed event is rejected and logged without losing the
rest of the batch.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from egovision.studio.settings import ACTIONS

WORKER_ID_PATTERN = r"^W[1-9][0-9]{0,2}$"
Verification = Literal["verified", "corrected", "low_confidence", "unverified"]
Completion = Literal["completed", "in_progress", "incomplete"]
VideoStatus = Literal["uploaded", "queued", "processing", "completed", "failed", "cancelled", "interrupted"]


def _clean_text(v: Any, limit: int) -> Any:
    return " ".join(v.split())[:limit] if isinstance(v, str) else v


class ReplyWorker(BaseModel):
    model_config = ConfigDict(extra="ignore")

    worker_id: str = Field(pattern=WORKER_ID_PATTERN)
    description: str = Field(min_length=1, max_length=200)
    is_camera_wearer: bool = False

    @field_validator("description", mode="before")
    @classmethod
    def _desc(cls, v: Any) -> Any:
        return _clean_text(v, 200)


class ReplyEvent(BaseModel):
    """One observed task as the analyser reports it. Times are timestamps of sampled frames (seconds)."""

    model_config = ConfigDict(extra="ignore")

    continues_event_id: str | None = None
    worker_id: str = Field(pattern=WORKER_ID_PATTERN)
    action: str
    object: str = Field(min_length=1, max_length=80)
    tool: str | None = Field(None, max_length=80)
    start_ts: float = Field(ge=0)
    end_ts: float = Field(ge=0)
    status: Literal["completed", "in_progress", "interrupted"]
    confidence: float = Field(ge=0, le=1)
    sop_step: int | None = Field(None, ge=1)
    evidence: str = Field("", max_length=300)

    @field_validator("object", "tool", mode="before")
    @classmethod
    def _names(cls, v: Any) -> Any:
        return _clean_text(v, 80).lower() if isinstance(v, str) else v

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, v: Any) -> Any:
        return _clean_text(v, 300)

    @field_validator("action", mode="before")
    @classmethod
    def _action(cls, v: Any) -> Any:
        if not isinstance(v, str) or v.strip().lower() not in ACTIONS:
            raise ValueError(f"action {v!r} is not in the fixed vocabulary")
        return v.strip().lower()

    @field_validator("tool", mode="after")
    @classmethod
    def _no_tool(cls, v: str | None) -> str | None:
        return None if v in (None, "", "none", "null", "n/a", "hands", "bare hands") else v

    @model_validator(mode="after")
    def _ordered(self) -> ReplyEvent:
        if self.end_ts < self.start_ts:
            raise ValueError(f"end_ts ({self.end_ts}) is before start_ts ({self.start_ts})")
        return self


class BatchReply(BaseModel):
    """Envelope of one reply; events are validated one by one afterwards."""

    model_config = ConfigDict(extra="ignore")

    workers: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    notes: str = ""


# JSON schema sent as Gemini's response_json_schema, so the reply is parseable JSON of this shape. Numeric
# ranges and patterns are enforced by the Pydantic models above (the backend never trusts the reply).
def reply_schema() -> dict[str, Any]:
    nullable_str = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["workers", "events", "notes"],
        "properties": {
            "workers": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["worker_id", "description", "is_camera_wearer"],
                "properties": {"worker_id": {"type": "string"}, "description": {"type": "string"},
                               "is_camera_wearer": {"type": "boolean"}}}},
            "events": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["continues_event_id", "worker_id", "action", "object", "tool", "start_ts", "end_ts",
                             "status", "confidence", "sop_step", "evidence"],
                "properties": {
                    "continues_event_id": nullable_str,
                    "worker_id": {"type": "string"},
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "object": {"type": "string"},
                    "tool": nullable_str,
                    "start_ts": {"type": "number"},
                    "end_ts": {"type": "number"},
                    "status": {"type": "string", "enum": ["completed", "in_progress", "interrupted"]},
                    "confidence": {"type": "number"},
                    "sop_step": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "evidence": {"type": "string"},
                }}},
            "notes": {"type": "string"},
        },
    }


class EventRecord(BaseModel):
    """A validated event as stored in SQLite and sent to the browser."""

    event_id: str
    video_id: str
    seq: int
    worker_id: str
    action: str
    object: str
    tool: str | None = None
    start_ts: float = Field(ge=0)
    end_ts: float = Field(ge=0)
    duration_s: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    verification_status: Verification
    completion_status: Completion
    frames_seen: int = Field(ge=1)
    sop_step: int | None = None
    evidence: str = ""
    corrections: list[str] = Field(default_factory=list)
    source: str = ""  # gemini:<model> | offline:forge | import:forge
    first_batch: int
    last_batch: int
    created_at: float
    updated_at: float


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    restart: bool = Field(False, description="Delete stored events and start over instead of resuming.")
    analyzer: Literal["gemini", "offline"] | None = Field(None, description="None: the server default.")
    max_seconds: float | None = Field(None, gt=0, description="Only analyse the first N seconds.")
    sop: list[str] | None = Field(None, description="Standard operating procedure, one step per item.")
    sample_every_s: float | None = Field(None, gt=0, le=30)
    batch_frames: int | None = Field(None, ge=2, le=40)

    @field_validator("sop", mode="before")
    @classmethod
    def _sop(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.splitlines()
        if isinstance(v, list):
            v = [" ".join(str(s).split())[:160] for s in v if str(s).strip()]
            if len(v) > 50:
                raise ValueError("an SOP can have at most 50 steps")
        return v  # [] = explicitly no SOP; None = keep the SOP given at upload
