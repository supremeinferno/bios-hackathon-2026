"""Offline events from the FORGE pipeline: no API, runs on the local GPU.

Used (1) when Gemini is not configured or fails, for the part of the video Gemini did not cover, (2) when the
user picks the offline analyser, and (3) to import results FORGE already produced (outputs/<clip>/result.json).
FORGE follows the camera wearer's hands only, so every offline event belongs to worker W1. Its whole-clip stages
(hands, objects, steps) mean events arrive together at the end, not batch by batch. Measured accuracy is far
below Gemini's (docs/RESULTS.md), which the UI states next to these events.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from egovision.schema import Action
from egovision.steps.postprocess import canonical_object
from egovision.studio.models import EventRecord
from egovision.studio.settings import ACTIONS, StudioSettings

WEARER = "camera wearer (the offline labeller follows only the wearer's hands)"
SYNONYMS: dict[str, tuple[str, ...]] = {
    "pick": ("take", "grab", "get", "lift", "collect"), "place": ("put", "set", "position", "lay", "load"),
    "fasten": ("screw", "bolt", "tighten", "rivet"), "inspect": ("check", "examine", "verify", "look"),
    "attach": ("fit", "mount", "clip", "tape", "assemble"), "insert": ("feed",), "press": ("push", "button"),
    "wipe": ("clean",), "measure": ("gauge",), "carry": ("move", "transport", "bring"), "cut": ("trim", "saw"),
    "adjust": ("align",), "operate": ("run", "start"), "write": ("mark", "label", "scan"),
}


def forge_result(s: StudioSettings, filename: str, duration_s: float) -> dict[str, Any] | None:
    """FORGE results for the same clip (same file name stem and length), or None."""
    d = s.forge_outputs_dir / Path(filename).stem
    if not (d / "result.json").is_file():
        return None
    r = json.loads((d / "result.json").read_text(encoding="utf-8"))
    run = json.loads((d / "run.json").read_text(encoding="utf-8")) if (d / "run.json").is_file() else {}
    full = float(run.get("video", {}).get("duration_s") or r["video"]["duration_s"])
    if abs(full - duration_s) > 0.5:
        return None
    san = r.get("files", {}).get("sanitized_video")
    return {"actions": [Action.model_validate(a) for a in r["actions"]],
            "processed_until_s": float(run.get("processed_until_s") or r["video"]["duration_s"]),
            "steps_backend": run.get("stats", {}).get("steps_backend") or "forge",
            "sanitized": san if san and Path(san).is_file() else None, "dir": d}


def run_forge(s: StudioSettings, video: dict[str, Any], limit_s: float | None,
              progress: Callable[[str, float], None] | None = None) -> tuple[list[Action], float]:
    """Run the FORGE pipeline with the offline labeller on this video. Outputs stay under the video's folder."""
    from egovision.config import Config
    from egovision.pipeline import run_video

    ov = ["steps.backend=exemplar", "privacy.enabled=false", f"ingest.stride={s.offline_stride}"]
    if limit_s:
        ov.append(f"ingest.max_seconds={limit_s}")
    res = run_video(Path(video["stored_path"]), s.outputs_dir / video["video_id"] / "forge", Config.load(None, ov),
                    render=False, progress=progress)
    return res.actions, res.meta.duration_s


def match_sop(action: str, obj: str, sop: list[str]) -> int | None:
    """SOP step whose text names this verb (or a synonym), preferring steps that also name the object."""
    words_obj = set(canonical_object(obj).split()) - {"unknown"}
    best, best_score = None, 0
    for i, step in enumerate(sop, 1):
        words = {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
                 for w in re.findall(r"[a-z]+", step.lower())}
        direct = action in words
        if not direct and not words & set(SYNONYMS.get(action, ())):
            continue
        score = (3 if direct else 2) + len(words_obj & words)  # the verb itself beats a synonym ("screw")
        if score > best_score:
            best, best_score = i, score
    return best


def events_from_actions(actions: list[Action], video: dict[str, Any], s: StudioSettings, first_seq: int, from_s: float,
                        end_s: float, source: str) -> list[EventRecord]:
    """FORGE actions (from `from_s` on) as Studio events of worker W1. Rejected actions are left out."""
    every = float(video.get("sample_every_s") or s.sample_every_s)
    sop = video.get("sop") or []
    now, out, seq = time.time(), [], first_seq
    for a in sorted(actions, key=lambda a: (a.start_s, a.end_s)):
        start, end = max(a.start_s, from_s), min(a.end_s, end_s)
        if a.status == "rejected" or end - start < 0.2 or a.verb not in ACTIONS:
            continue
        conf = a.confidence if a.confidence is not None else {"verified": 0.75, "corrected": 0.65}.get(a.status or "", 0.5)
        notes = [f"verification by FORGE: {a.status}"] if a.status == "corrected" else []
        if start > a.start_s:
            notes.append(f"started at {a.start_s:.1f} s; only the part after {from_s:.1f} s is counted here")
        step = match_sop(a.verb, a.object, sop) if sop else None
        if step is not None:
            notes.append("SOP step matched by keywords (offline labeller)")
        status = "low_confidence" if conf < s.min_confidence else {"verified": "verified", "corrected": "corrected"}.get(
            a.status or "", "unverified")
        evidence = f"{a.hand} hand · {a.source or 'forge'}" + (f" · flags: {', '.join(a.flags)}" if a.flags else "")
        out.append(EventRecord(
            event_id=f"E{seq:04d}", video_id=video["video_id"], seq=seq, worker_id="W1", action=a.verb,
            object=(a.object or "unknown").lower()[:80], tool=None, start_ts=round(start, 3), end_ts=round(end, 3),
            duration_s=round(end - start, 3), confidence=round(conf, 3), verification_status=status,  # type: ignore[arg-type]
            completion_status="incomplete" if end >= end_s - 0.5 else "completed",
            frames_seen=max(1, round((end - start) / every)), sop_step=step, evidence=evidence, corrections=notes,
            source=source, first_batch=-1, last_batch=-1, created_at=now, updated_at=now))
        seq += 1
    return out
