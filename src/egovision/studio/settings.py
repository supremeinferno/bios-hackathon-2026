"""Studio settings. Defaults below; any field can be overridden with an env var `STUDIO_<FIELD>`
(e.g. `STUDIO_GEMINI_MODEL=gemini-flash-lite-latest`, `STUDIO_SAMPLE_EVERY_S=2`; lists comma-separated), also
read from `.env`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Closed action vocabulary: every event the analyser returns must use one of these (anything else is rejected).
# The hand verbs match the FORGE list (config.DEFAULT_VERBS); measure/operate/walk/wait are added because
# worker analytics needs travel and waiting time, which the hand-only list cannot express.
ACTIONS: dict[str, str] = {
    "pick": "take an object in the hand(s)",
    "place": "put an object down at a destination",
    "hold": "keep an object still while something else happens",
    "carry": "move an object while walking or turning",
    "insert": "put one part into another (peg, screw into hole, part into fixture)",
    "attach": "join parts without a tool: clips, end caps, tape, labels",
    "fasten": "screw, bolt, rivet or tighten with a hand or a tool",
    "cut": "cut, trim or saw",
    "sand": "sand, plane or file a surface",
    "grind": "grind or polish with a machine",
    "wipe": "clean or wipe a surface",
    "push": "push, slide, tilt or pull a large object",
    "press": "press a button, pedal, lever or press a part into place",
    "adjust": "align, reposition or fine-tune a part or a machine setting",
    "inspect": "look at or check a part closely",
    "measure": "use a ruler, gauge, tape or caliper",
    "operate": "run a machine or power tool that does the work",
    "write": "write, mark or scan a label or form",
    "walk": "move between work positions without carrying anything",
    "wait": "idle or waiting: hands free, no task in progress",
    "other": "a work action none of the above describes",
}



class StudioSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ---- storage (all git-ignored: data/ and outputs*/) -----------------------------------------
    videos_dir: Path = Field(Path("data/videos"), description="Uploaded originals + browser preview copies.")
    outputs_dir: Path = Field(Path("outputs/videos"), description="Per-video reports and raw analyser replies.")
    db_path: Path = Field(Path("data/studio.sqlite3"), description="SQLite database (source of truth).")
    sanitized_dir: Path = Field(Path("data/sanitized"), description="FORGE sanitized copies, preferred as frame source.")
    max_upload_mb: int = Field(4096, ge=1)
    forge_outputs_dir: Path = Field(Path("outputs"), description="FORGE pipeline results (<clip>/result.json).")
    forge_video_dirs: list[Path] = Field(default_factory=lambda: [Path("data/raw"), Path("data/uploads")],
                                         description="Where FORGE's clips live (for importing their results).")

    # ---- analyser ---------------------------------------------------------------------------------
    analyzer: Literal["gemini", "offline"] = Field(
        "gemini", description="gemini = live batches via the Gemini API, with the offline FORGE labeller as fallback; "
                              "offline = FORGE labeller only (local GPU, no API, events arrive at the end).")
    gemini_model: str = Field("gemini-3.5-flash", description="Same main model as FORGE's measured Gemini step.")
    gemini_fallback_models: list[str] = Field(default_factory=lambda: ["gemini-flash-lite-latest", "gemini-3.8-flash"])
    max_output_tokens: int = Field(16000, ge=1024)
    api_timeout_s: float = Field(120.0, gt=0, description="Per-request timeout; a hung call moves to the next model.")
    max_retries: int = Field(2, ge=0, le=10, description="Retries per model on overload (503) / timeouts.")
    max_failed_batches: int = Field(3, ge=1, description="This many failed batches in a row -> offline fallback.")
    offline_stride: int = Field(2, ge=1, le=6, description="FORGE hand-model frame stride for the offline labeller.")
    hands3d_hfov_deg: float = Field(150.0, gt=60, lt=220, description="Assumed fisheye field of view for 3D hands "
                                                                       "(no intrinsics exist; distances scale with it).")

    # ---- frame sampling -------------------------------------------------------------------------
    sample_every_s: float = Field(1.0, gt=0, le=30, description="One frame every N seconds is sent to Gemini.")
    batch_frames: int = Field(10, ge=2, le=40, description="Frames per Gemini request.")
    image_width: int = Field(768, ge=256, le=1568, description="Frames are resized to this width before upload.")
    jpeg_quality: int = Field(80, ge=40, le=100)
    face_blur: bool = Field(True, description="Pixelate faces (OpenCV Haar, best effort) before frames leave the machine.")
    preview_width: int = Field(960, ge=320, description="Width of the H.264 browser preview made for non-H.264 uploads.")

    # ---- verification + analytics ---------------------------------------------------------------
    min_confidence: float = Field(0.6, ge=0, le=1, description="Below this an event is 'low_confidence'.")
    snap_tolerance_s: float = Field(0.25, ge=0, description="Timestamps further than this from a sampled frame are corrected.")
    long_factor: float = Field(2.0, gt=1, description="Unusually long: duration >= factor x median of the same task ...")
    long_min_excess_s: float = Field(5.0, ge=0, description="... and at least this many seconds above that median.")
    long_min_samples: int = Field(3, ge=2, description="Other events of the same task needed for a median baseline.")
    long_fallback_s: float = Field(90.0, gt=0, description="Without any baseline, a task this long is flagged.")
    bottleneck_share: float = Field(0.25, gt=0, le=1, description="A task taking this share of observed time is a bottleneck.")
    wait_share: float = Field(0.10, gt=0, le=1, description="Waiting above this share of a worker's time is flagged.")

    @classmethod
    def from_env(cls, **overrides: object) -> StudioSettings:
        try:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True), override=False)
        except ImportError:
            pass
        vals: dict[str, object] = {}
        for name in cls.model_fields:
            env = os.environ.get(f"STUDIO_{name.upper()}")
            if env not in (None, ""):
                vals[name] = [x.strip() for x in env.split(",") if x.strip()] if name in (
                    "gemini_fallback_models", "forge_video_dirs") else env
        vals.update(overrides)
        return cls.model_validate(vals)
