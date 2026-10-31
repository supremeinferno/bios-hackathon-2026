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
