"""Data contracts between stages and for exported files.

`hands.jsonl` follows the PS1 suggested format (PDF p.3) plus `id` and `interp`;
`actions.csv` follows the PS3 suggested format (PDF p.4) exactly.
Coordinates are pixels in the ORIGINAL video resolution, whatever resolution a model ran at.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Side = Literal["left", "right"]
HandUse = Literal["left", "right", "both", "none"]
ACTIONS_CSV_COLUMNS = ["clip", "start_s", "end_s", "verb", "object", "hand"]


class VideoMeta(BaseModel):
    path: str
    name: str  # file stem, used as `clip` in outputs
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    codec: str


class Hand(BaseModel):
    id: int | None = None
    side: Side
    score: float = Field(ge=0, le=1)
    bbox: list[float]  # [x1, y1, x2, y2]
    kp2d: list[list[float]]  # 21 x [x, y]
    interp: bool = False  # True if filled between processed frames, not detected

    @field_validator("bbox")
    @classmethod
    def _bbox4(cls, v: list[float]) -> list[float]:
        if len(v) != 4 or v[2] < v[0] or v[3] < v[1]:
            raise ValueError(f"bbox must be [x1,y1,x2,y2] with x2>=x1, y2>=y1, got {v}")
        return v

    @field_validator("kp2d")
    @classmethod
    def _kp21(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 21 or any(len(p) != 2 for p in v):
            raise ValueError("kp2d must be 21 [x, y] points")
        return v


class FrameHands(BaseModel):
    frame: int = Field(ge=0)
    t: float = Field(ge=0)
    hands: list[Hand] = Field(default_factory=list)


ContactState = Literal["none", "portable", "fixed"]
CONTACTS_CSV_COLUMNS = ["clip", "t_start", "t_end", "hand", "state", "object_name"]


class ObjectDet(BaseModel):
    label: str
    kind: Literal["portable", "fixed", "privacy"]
    score: float = Field(ge=0, le=1)
    bbox: list[float]  # original pixels


class Contact(BaseModel):
    """What one tracked hand touches at one keyframe (PS2 definition, PDF p.4)."""

    hand_id: int
    side: Side
    state: ContactState
    object: str | None = None
    object_bbox: list[float] | None = None
    object_score: float | None = None
    n_tips: int = 0  # fingertips inside the padded object box


class FrameObjects(BaseModel):
    frame: int
    t: float
    objects: list[ObjectDet] = Field(default_factory=list)
    contacts: list[Contact] = Field(default_factory=list)


class ContactSegment(BaseModel):
    clip: str
    t_start: float
    t_end: float
    hand: Side
    state: ContactState
    object_name: str

    def csv_row(self) -> list[str]:
        return [self.clip, f"{self.t_start:.2f}", f"{self.t_end:.2f}", self.hand, self.state, self.object_name]


class Action(BaseModel):
    clip: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    verb: str
    object: str
    hand: HandUse
    confidence: float | None = Field(None, ge=0, le=1)
    source: str = ""  # which backend produced it

    @model_validator(mode="after")
    def _ordered(self) -> Action:
        if self.end_s <= self.start_s:
            raise ValueError(f"end_s ({self.end_s}) must be > start_s ({self.start_s})")
        return self

    def csv_row(self) -> list[str]:
        return [self.clip, f"{self.start_s:.2f}", f"{self.end_s:.2f}", self.verb, self.object, self.hand]
