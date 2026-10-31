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
    "sand", "grind", "wipe", "push", "press", "adjust", "inspect", "write", "other",
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


class QCConfig(_Strict):
    """Stage 0 quality control (egovision/qc.py)."""

    enabled: bool = True
    size: int = Field(320, ge=64, description="Width of the grey copy used for sharpness/brightness.")
    blur_rel_threshold: float = Field(0.35, gt=0, lt=1, description="Blurry if sharpness < this x clip median (D2).")
    covered_max_brightness: float = Field(20.0, ge=0, description="Mean grey level below this = lens covered/dark.")
    covered_max_std: float = Field(6.0, ge=0, description="Grey std below this = featureless (lens covered).")


class SignalsConfig(_Strict):
    """Stage 4 camera-origin coordinates and hand signals (egovision/signals.py)."""

    enabled: bool = True
    hz: float = Field(10.0, gt=0, le=30, description="Sampling rate of signals.csv.")


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
    # Stage 3 additions (hands/mediapipe_backend.py; measured in docs/DECISIONS.md D12)
    # Both OFF after measurement (DECISIONS D12): recovery +0-1 pt presence at half the speed and added
    # false hands (rollers, feet); the coworker rule dropped the wearer's own hands.
    recover: bool = False
    recover_clahe_clip: float = Field(3.0, gt=0, description="CLAHE clip limit for the retry copy.")
    recover_min_score: float = Field(0.6, ge=0, le=1, description="Min handedness score for a recovered hand.")
    filter_other_person: bool = False
    other_person_min_down: float = Field(0.25, ge=0, description="Wrist->knuckle downward extent / hand size.")
    other_person_max_wrist_y: float = Field(0.55, gt=0, le=1, description="Only hands with wrist above this frame share.")
    glove_threshold_drop: float = Field(0.15, ge=0, le=0.5, description="Lower detection threshold when gloved.")
    # Linear interpolation across skipped frames only if the gap is at most this many source frames.
    max_interp_gap: int = Field(6, ge=0)
    # Tracking (hands/tracking.py)
    track_max_age_s: float = Field(0.4, ge=0, description="Keep an unmatched track active this long.")
    reid_window_s: float = Field(3.0, ge=0, description="A re-entering hand of the same side resumes a lost ID.")
    track_max_cost: float = Field(1.6, gt=0, description="Max (1-IoU + norm. centre distance + side penalty).")
    track_side_penalty: float = Field(0.5, ge=0)
    side_vote: bool = True
    # One-Euro smoothing (PDF step 4)
    smooth: bool = True
    smooth_min_cutoff: float = Field(1.5, gt=0)
    smooth_beta: float = Field(0.01, ge=0)


class VocabItem(_Strict):
    name: str  # canonical object name used in outputs
    prompt: str  # text sent to the open-vocabulary detector
    kind: Literal["portable", "fixed", "privacy"] = "portable"
    min_score: float | None = Field(None, ge=0, le=1, description="Per-item threshold (default: global).")


def _v(name: str, prompt: str | None = None, kind: str = "portable") -> VocabItem:
    return VocabItem(name=name, prompt=prompt or name, kind=kind)  # type: ignore[arg-type]


# Object names come from what is visible in each recording source (docs/DATA_REPORT.md §4).
# Keys are clip-name prefixes; clips from unknown sources use the union of all lists.
DEFAULT_VOCAB: dict[str, list[VocabItem]] = {
    "MTL-SGF00106": [
        _v("plywood board"), _v("wooden strip"), _v("power drill"), _v("electric planer", "electric hand planer"),
        _v("jigsaw"), _v("angle grinder"), _v("hammer"), _v("pencil"),
        _v("furniture frame", "wooden furniture frame", "fixed"),
    ],
    "MTL-SKL00089": [
        _v("aluminium profile", "long aluminium bar"), _v("diffuser strip", "long white plastic strip"),
        _v("circuit board", "small green circuit board"), _v("end cap", "small grey plastic cap"),
        _v("tape roll"), _v("pen"), _v("paper sheet"), _v("plastic bag"), _v("scissors"),
        _v("workbench", "table", "fixed"),
    ],
    "mytronlabs-000003": [
        _v("metal part", "small metal part"), _v("clamp", "hand clamp tool"), _v("file", "metal file"),
        _v("screwdriver"), _v("mobile phone"), _v("cloth"), _v("plastic comb"),
        _v("grinding wheel", "grinding wheel", "fixed"), _v("machine", "metal machine table", "fixed"),
    ],
    "mytronlabs-000060": [
        _v("cloth", "cloth rag"), _v("hand tool", "yellow hand tool"), _v("metal bar"), _v("hand truck"),
        _v("granite slab", "granite slab", "fixed"), _v("roller conveyor", "roller conveyor", "fixed"),
        _v("slab clamp", "yellow lifting clamp", "fixed"),
    ],
}


def _p(category: str, prompt: str, min_score: float) -> VocabItem:
    return VocabItem(name=f"privacy:{category}", prompt=prompt, kind="privacy", min_score=min_score)


# Stage 1 privacy prompts for the same OWLv2 pass: faces and company names only. "person" finds
# nearly every coworker and "face" alone misses most, so heads are blurred from person boxes too
# (docs/DECISIONS.md D9). Text/document/screen prompts were dropped because they blurred the
# workers' tools and parts (D11).
DEFAULT_PRIVACY_PROMPTS: list[VocabItem] = [
    _p("person", "person", 0.15), _p("head", "human head", 0.15), _p("face", "human face", 0.15),
    _p("sign", "signboard with company name", 0.15), _p("logo", "company logo", 0.2),
]


class PrivacyConfig(_Strict):
    """Stage 1: blur faces (people's heads) and company names (signs, logos) in a sanitized copy.

    The sanitized copy is what goes to cloud APIs, the UI and demos. Hand tracking runs on originals.
    Blurring is an extra safety layer; it does NOT make footage publishable (PDF data-use rules).
    """

    enabled: bool = True
    prompts: list[VocabItem] = Field(default_factory=lambda: list(DEFAULT_PRIVACY_PROMPTS))
    person_head_frac: float = Field(0.25, gt=0, le=1, description="Top share of a person box treated as head.")
    pad_frac: float = Field(0.1, ge=0, description="Pad every region by this share of its size.")
    hold_s: float = Field(0.35, ge=0, description="Apply each detection to frames within +/- this many seconds.")
    drop_wearer_arms: bool = Field(True, description="Ignore 'person' boxes that are the wearer's own arms.")
    head_max_top_frac: float = Field(0.7, gt=0, le=1, description=(
        "Drop head/face regions whose top edge is below this share of the frame height: in head-camera "
        "video those are the wearer's own knees/feet/body (docs/DECISIONS.md D10)."))
    protect_hands: bool = True
    hand_pad_frac: float = Field(0.08, ge=0)
    pixel_block: int = Field(14, ge=4, description="Pixelation block size in output pixels (bigger = stronger).")
    output_width: int = Field(960, ge=160)
    out_dir: Path = Path("data/sanitized")


class ObjectsConfig(_Strict):
    enabled: bool = True
    model: str = "google/owlv2-base-patch16-ensemble"
    every_s: float = Field(0.33, gt=0, description="Run the detector on one processed frame per this many seconds.")
    score_threshold: float = Field(0.2, ge=0, le=1)
    nms_iou: float = Field(0.5, gt=0, le=1)
    max_area_frac: float = Field(0.35, gt=0, le=1, description="Drop portable boxes larger than this frame share.")
    batch_size: int = Field(4, ge=1)
    vocab: dict[str, list[VocabItem]] = Field(default_factory=lambda: dict(DEFAULT_VOCAB))
    # Contact rule (PDF step 5): >= min_tips fingertips inside the object box padded by margin * hand size.
    contact_margin: float = Field(0.15, ge=0)
    contact_min_tips: int = Field(2, ge=1, le=5)
    contact_min_s: float = Field(0.4, ge=0, description="Contacts shorter than this are noise (PDF p.12).")

    def vocab_for(self, clip: str) -> list[VocabItem]:
        keys = [k for k in self.vocab if k != "default" and clip.startswith(k)]
        if keys:
            return self.vocab[max(keys, key=len)]
        if "default" in self.vocab:
            return self.vocab["default"]
        seen: dict[str, VocabItem] = {}
        for items in self.vocab.values():
            for it in items:
                seen.setdefault(it.name, it)
        return list(seen.values())


# Cue prototypes for the exemplar matcher: how strongly a held/touched object suggests each verb.
DEFAULT_OBJECT_VERBS: dict[str, dict[str, float]] = {
    "power drill": {"fasten": 1.0}, "electric planer": {"sand": 1.0}, "angle grinder": {"grind": 0.7, "sand": 0.5},
    "jigsaw": {"cut": 1.0}, "hammer": {"fasten": 1.0}, "pencil": {"write": 1.0}, "pen": {"write": 1.0},
    "paper sheet": {"write": 0.6, "inspect": 0.4}, "plywood board": {"hold": 0.5, "carry": 0.4, "place": 0.3},
    "wooden strip": {"hold": 0.4, "attach": 0.4, "place": 0.3}, "furniture frame": {"attach": 0.5, "adjust": 0.4},
    "aluminium profile": {"insert": 0.6, "hold": 0.4, "inspect": 0.3},
    "diffuser strip": {"insert": 0.8, "hold": 0.3}, "circuit board": {"attach": 0.5, "insert": 0.4, "inspect": 0.3},
    "end cap": {"attach": 1.0}, "tape roll": {"attach": 0.8}, "scissors": {"cut": 1.0}, "plastic bag": {"pick": 0.5},
    "metal part": {"grind": 0.5, "adjust": 0.4, "inspect": 0.4}, "clamp": {"adjust": 0.7, "fasten": 0.4},
    "file": {"grind": 0.8}, "screwdriver": {"fasten": 1.0}, "mobile phone": {"other": 0.6, "inspect": 0.3},
    "grinding wheel": {"grind": 1.0}, "machine": {"adjust": 0.4, "push": 0.3}, "cloth": {"wipe": 1.0},
    "granite slab": {"push": 0.6, "wipe": 0.4, "hold": 0.3}, "roller conveyor": {"push": 0.7},
    "slab clamp": {"adjust": 0.6, "push": 0.3}, "hand tool": {"press": 0.6, "adjust": 0.4},
    "metal bar": {"push": 0.5, "carry": 0.5},
    "hand truck": {"push": 1.0},
    "workbench": {"other": 0.3},
}


# Text prototypes per verb for SigLIP ("a first-person photo of hands <phrase>"); {o} = object name.
DEFAULT_VERB_PHRASES: dict[str, list[str]] = {
    "pick": ["picking up {o}", "grabbing {o}"],
    "place": ["putting down {o}", "placing {o} on a surface"],
    "hold": ["holding {o}", "holding {o} still"],
    "carry": ["carrying {o} while walking"],
    "insert": ["inserting {o} into another part", "sliding {o} into a channel"],
    "attach": ["attaching {o} onto another part", "fitting {o} on the end of a part", "taping {o}"],
    "fasten": ["fastening {o} with a power drill", "screwing {o}", "nailing {o}"],
    "cut": ["cutting {o} with a saw", "cutting {o} with scissors"],
    "sand": ["sanding {o} with a power sander", "planing {o}"],
    "grind": ["grinding {o} on a grinding wheel", "polishing {o}", "filing {o}"],
    "wipe": ["wiping {o} with a cloth", "cleaning {o} with a rag"],
    "push": ["pushing {o}", "sliding {o} along rollers"],
    "press": ["pressing buttons on {o}", "pressing {o}"],
    "adjust": ["adjusting {o}", "tightening {o}", "aligning {o}"],
    "inspect": ["inspecting {o} closely", "looking at {o} in the hands"],
    "write": ["writing on {o} with a pen"],
    "other": ["hands doing something with {o}"],
}


class StepsConfig(_Strict):
    backend: Literal["exemplar", "gemini"] = Field(
        "exemplar", description="exemplar = offline prototype matching on hand/object/visual cues; gemini = API."
    )
    verbs: list[str] = Field(default_factory=lambda: list(DEFAULT_VERBS))
    # Exemplar matcher
    embed_model: str = "google/siglip-base-patch16-224"
    crop_pad: float = Field(0.35, ge=0, description="Hand(+object) crop padding as a fraction of its size.")
    crop_min_px: int = Field(96, ge=16)
    exemplars_file: Path | None = Field(None, description="Optional labelled exemplars (few-shot visual prototypes).")
    object_verbs: dict[str, dict[str, float]] = Field(default_factory=lambda: dict(DEFAULT_OBJECT_VERBS))
    w_text: float = Field(1.0, ge=0, description="Weight of SigLIP text-prototype similarity.")
    w_cue: float = Field(1.0, ge=0, description="Weight of held-object cue prior.")
    w_visual: float = Field(1.5, ge=0, description="Weight of visual exemplar similarity (if exemplars exist).")
    text_temperature: float = Field(0.05, gt=0)
    idle_speed: float = Field(0.25, ge=0, description="Hand speed (hand-sizes/s) below which an empty hand is idle.")
    verb_phrases: dict[str, list[str]] = Field(default_factory=lambda: dict(DEFAULT_VERB_PHRASES))
    idle_prob_empty_slow: float = Field(0.7, gt=0, lt=1, description="P(idle) for an empty, still hand.")
    idle_prob_empty_moving: float = Field(0.4, gt=0, lt=1)
    idle_prob_contact: float = Field(0.1, gt=0, lt=1)
    absent_cost: float = Field(10.0, ge=0, description="Emission cost of any verb when the hand is not visible.")
    speed_lag_s: float = Field(0.2, gt=0, description="Time lag for hand speed estimation.")
    switch_penalty: float = Field(2.0, ge=0, description="Viterbi cost for changing label between keyframes.")
    min_segment_s: float = Field(1.0, ge=0)
    pick_place_s: float = Field(0.8, ge=0, description="Length of pick/place carved at grab/release (0 = off).")
    both_iou: float = Field(0.5, ge=0, le=1, description="Merge L+R segments with same verb/object into 'both'.")
    # Gemini (optional; PDF minimum goal)
    gemini_model: str = "gemini-3.5-flash"  # measured: 10 s window in 3.6 s (DECISIONS D14)
    gemini_fallback_models: list[str] = Field(
        default_factory=lambda: ["gemini-flash-lite-latest", "gemini-3.8-flash"],
        description="Tried in order when the main model is overloaded (503) or times out.")
    request_timeout_s: float = Field(300.0, gt=0, description="Per-request timeout; a hung call fails, then falls back.")
    api_key_env: str = "GEMINI_API_KEY"
    video_fps: float = Field(2.0, gt=0, le=24, description="Frames/s Gemini samples from the video (pass 1).")
    window_s: float = Field(30.0, gt=1, description="Pass 1 window length; whole clips were refused (503).")
    stitch_gap_s: float = Field(1.0, ge=0, description="Join same verb/object/hand actions across window edges.")
    pass2: bool = False  # hurt on the dev clip (F1@10 35.3 -> 16.7); DECISIONS D15
    pass2_fps: float = Field(4.0, gt=0, le=24, description="Higher sampling for pass-2 windows.")
    pass2_window_s: float = Field(12.0, gt=1)
    pass2_min_confidence: float = Field(0.5, ge=0, le=1, description="Re-ask windows around actions below this.")
    pass2_max_windows: int = Field(6, ge=0)
    upload_width: int = Field(854, ge=160, description="Clip is downscaled to this width before upload.")
    max_retries: int = Field(2, ge=0, description="Retries per model before moving to the next model.")
    cache_dir: Path = Path("cache/api")
    delete_uploads: bool = True  # data-use rule: remove uploaded clips from API storage
    allow_unsanitized_upload: bool = Field(False, description="FORGE Stage 1: only sanitized copies go to the API.")


class VerifyConfig(_Strict):
    """Stages 7-8: hand-grounded verification and boundary snapping (egovision/verify.py)."""

    enabled: bool = True
    min_presence: float = Field(0.3, ge=0, le=1, description="Claimed hand must be visible this share of the action.")
    reject_presence: float = Field(0.05, ge=0, le=1, description="Below this for every hand -> rejected.")
    min_activity: float = Field(0.02, ge=0, description="Below this, attribution is not attempted.")
    both_ratio: float = Field(2.0, ge=1, description="Activity ratio under which both hands count as active.")
    still_verbs: list[str] = Field(default_factory=lambda: ["hold", "inspect"])
    still_max_speed: float = Field(0.6, ge=0, description="Camera-units/s; above this a 'still' verb is suspect.")
    moving_verbs: list[str] = Field(default_factory=lambda: ["wipe", "push", "carry", "sand", "grind"])
    moving_min_speed: float = Field(0.03, ge=0)
    snap: bool = True
    snap_window_s: float = Field(1.2, ge=0, description="Search +/- this for a motion valley (FORGE: 1-1.5 s).")
    min_len_after_snap_s: float = Field(0.4, ge=0)
    snap_min_valid: float = Field(0.6, ge=0, le=1, description="Snap only if the hand has a speed in this share of the window.")


class SOPConfig(_Strict):
    """Stage 8 step order + completeness across clips of one source (egovision/sop.py)."""

    min_support: float = Field(0.5, gt=0, le=1, description="Keep steps seen in at least this share of clips.")
    min_clips: int = Field(2, ge=1)
    ignore_verbs: list[str] = Field(default_factory=lambda: ["other"])


class TrimConfig(_Strict):
    """Stage 10 smooth trim + compress (egovision/trim.py). Only after reviewer approval."""

    pad_s: float = Field(1.5, ge=0, description="Pre/post-roll around every kept action.")
    merge_gap_s: float = Field(4.0, ge=0, description="Kept segments closer than this are merged.")
    lite_fps: float = Field(1.0, gt=0, description="Frame rate kept outside kept segments (keep-lite).")
    width: int = Field(854, ge=160, description="Output width (480p).")
    codec: str = "libx265"
    crf: int = Field(30, ge=0, le=51)
    out_dir: Path = Path("data/trimmed")
    require_approval: bool = True


class RenderConfig(_Strict):
    enabled: bool = True
    width: int = Field(960, ge=160)
    timeline_height: int = Field(56, ge=0)
    draw_keypoints: bool = True
    draw_bbox: bool = True
    draw_objects: bool = True


class Config(_Strict):
    seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    profile_path: Path = Field(Path("config/worker_profiles.yaml"), description="Stage 2 worker profiles.")
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    hands: HandsConfig = Field(default_factory=HandsConfig)
    objects: ObjectsConfig = Field(default_factory=ObjectsConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    steps: StepsConfig = Field(default_factory=StepsConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    sop: SOPConfig = Field(default_factory=SOPConfig)
    trim: TrimConfig = Field(default_factory=TrimConfig)
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
