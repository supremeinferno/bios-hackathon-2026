"""Typed configuration. One `Config` object drives every stage; no thresholds live in stage code.

Load order: defaults below -> YAML file (optional) -> dotted CLI overrides (`hands.num_hands=2`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Fixed, closed verb list (SPEC.md A2). Every backend must map into this set.
DEFAULT_VERBS: list[str] = [
    "pick", "place", "hold", "carry", "insert", "attach", "fasten", "cut",
    "sand", "grind", "wipe", "push", "adjust", "inspect", "write", "other",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestConfig(_Strict):
    stride: int = Field(2, ge=1, description="Run per-frame models on every Nth frame; gaps are interpolated.")
    width: int = Field(960, ge=0, description="Processing width in px (0 = native). Aspect ratio is kept.")
    max_seconds: float | None = Field(None, gt=0, description="Only process the first N seconds (smoke tests).")
    cache_frames: bool = True
    cache_dir: Path = Path("cache/frames")
    jpeg_quality: int = Field(92, ge=50, le=100)
    min_duration_s: float = Field(1.0, gt=0, description="Reject clips shorter than this.")


class HandsConfig(_Strict):
    backend: Literal["mediapipe"] = "mediapipe"
    model_path: Path = Path("weights/hand_landmarker.task")
    num_hands: int = Field(2, ge=1, le=4)
    min_detection_confidence: float = Field(0.4, ge=0, le=1)
    min_presence_confidence: float = Field(0.4, ge=0, le=1)
    min_tracking_confidence: float = Field(0.4, ge=0, le=1)
    # PDF p.11 warns labels may come out mirrored. Measured on our footage with MediaPipe 1.2.0 they
    # do NOT: swapping put 79% of hands on the wrong half of the image (docs/DECISIONS.md D4).
    swap_handedness: bool = False
    # Linear interpolation across skipped frames only if the gap is at most this many source frames.
    max_interp_gap: int = Field(6, ge=0)


class ActionsConfig(_Strict):
    backend: Literal["auto", "gemini", "offline"] = Field(
        "auto", description="auto = gemini if the API key env var is set, else offline."
    )
    verbs: list[str] = Field(default_factory=lambda: list(DEFAULT_VERBS))
    # Gemini
    gemini_model: str = "gemini-3.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    video_fps: float = Field(2.0, gt=0, le=24, description="Frames/s Gemini samples from the video.")
    upload_width: int = Field(854, ge=160, description="Clip is downscaled to this width before upload.")
    request_timeout_s: float = 600.0
    max_retries: int = Field(4, ge=0)
    cache_dir: Path = Path("cache/api")
    delete_uploads: bool = True  # data-use rule: remove uploaded clips from API storage
    # Offline baseline (hand-activity segments)
    min_action_s: float = Field(0.8, gt=0)
    merge_gap_s: float = Field(0.5, ge=0)


class RenderConfig(_Strict):
    enabled: bool = True
    width: int = Field(960, ge=160)
    timeline_height: int = Field(56, ge=0)
    draw_keypoints: bool = True
    draw_bbox: bool = True


class Config(_Strict):
    seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    hands: HandsConfig = Field(default_factory=HandsConfig)
    actions: ActionsConfig = Field(default_factory=ActionsConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
        data: dict[str, Any] = {}
        if path is not None:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        for item in overrides or []:
            key, sep, raw = item.partition("=")
            if not sep:
                raise ValueError(f"override must look like section.key=value, got {item!r}")
            node = data
            parts = key.strip().split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = yaml.safe_load(raw)
        return cls.model_validate(data)
