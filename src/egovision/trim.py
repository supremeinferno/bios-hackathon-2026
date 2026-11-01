"""Stage 10 trim + compress (FORGE). Runs only after a reviewer approved the CURRENT trim plan.

Plan: every action except the ones a reviewer rejected, padded by `pad_s`. The gaps left between them
are idle; only idle stretches of at least `min_idle_s` are trimmed (shorter cuts make a jumpy video).
Stage 7 'rejected' actions are kept unless `drop_model_rejected`: on the stone-plant clip 10 of 17
rejected actions were real work done with the hands out of view (docs/RESULTS.md).

Two output modes, picked per run in review:
  cut        idle stretches are removed, so the video is shorter. trim_map.json holds the time map
             (trimmed time <-> original time); actions_trimmed.csv lists the actions in trimmed time.
  keep_lite  idle stretches stay at `lite_fps`: same length, every original timestamp still valid.
Input is the SANITIZED copy. Output is H.264 by default so browsers can play it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import av

from egovision.config import TrimConfig
from egovision.ingest import iter_all_frames
from egovision.schema import Action

Mode = Literal["cut", "keep_lite"]
MODES: tuple[Mode, ...] = ("cut", "keep_lite")


def kept_actions(actions: list[Action], cfg: TrimConfig) -> list[Action]:
    return [a for a in actions if not (a.review and a.review.get("status") == "rejected")
            and not (cfg.drop_model_rejected and a.status == "rejected")]


def plan(actions: list[Action], duration_s: float, cfg: TrimConfig) -> list[dict[str, Any]]:
    """Contiguous keep / idle segments covering [0, duration_s]."""
    spans = sorted((max(0.0, a.start_s - cfg.pad_s), min(duration_s, a.end_s + cfg.pad_s))
                   for a in kept_actions(actions, cfg))
    keep: list[list[float]] = []
    for s, e in spans:
        if keep and s - keep[-1][1] < cfg.min_idle_s:
            keep[-1][1] = max(keep[-1][1], e)
        else:
            keep.append([s, e])
    if keep and keep[0][0] < cfg.min_idle_s:
        keep[0][0] = 0.0  # a short idle lead-in is not worth a cut
    if keep and duration_s - keep[-1][1] < cfg.min_idle_s:
        keep[-1][1] = duration_s
    segs: list[dict[str, Any]] = []
    t = 0.0
    for s, e in keep:
        if s > t:
            segs.append({"start_s": round(t, 2), "end_s": round(s, 2), "mode": "idle"})
        segs.append({"start_s": round(s, 2), "end_s": round(e, 2), "mode": "keep"})
        t = e
    if t < duration_s:
        segs.append({"start_s": round(t, 2), "end_s": round(duration_s, 2), "mode": "idle"})
    return segs


def plan_sha(segs: list[dict[str, Any]]) -> str:
    """Fingerprint of a plan: an approval is valid only for the exact plan the reviewer saw."""
    return hashlib.sha1(json.dumps(segs, sort_keys=True).encode()).hexdigest()[:12]


def summary(segs: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    idle = [s for s in segs if s["mode"] == "idle"]
    idle_s = sum(s["end_s"] - s["start_s"] for s in idle)
    return {"cuts": len(idle), "idle_s": round(idle_s, 1), "kept_pct": round(100 * (1 - idle_s / max(duration_s, 1e-9)), 1),
            "cut_duration_s": round(duration_s - idle_s, 1), "sha": plan_sha(segs)}


# ---- frame selection and time map ------------------------------------------------------------------
def _first_frame(t: float, fps: float) -> int:
    return math.ceil(t * fps - 1e-6)


def time_map(segs: list[dict[str, Any]], fps: float) -> list[dict[str, float]]:
    """Cut mode: one entry per kept segment with its original and trimmed span (exact frame counts)."""
    out, n = [], 0
    for s in segs:
        if s["mode"] != "keep":
            continue
        a, b = _first_frame(s["start_s"], fps), _first_frame(s["end_s"], fps)
        if b <= a:
            continue
        out.append({"orig_start_s": round(a / fps, 3), "orig_end_s": round(b / fps, 3),
                    "trim_start_s": round(n / fps, 3), "trim_end_s": round((n + b - a) / fps, 3)})
        n += b - a
    return out


def to_trimmed(t: float, tmap: list[dict[str, float]]) -> float | None:
    """Original time -> trimmed time; None if t falls in a cut idle stretch."""
    for m in tmap:
        if m["orig_start_s"] <= t <= m["orig_end_s"]:
            return round(m["trim_start_s"] + t - m["orig_start_s"], 3)
    return None


def to_original(t: float, tmap: list[dict[str, float]]) -> float | None:
    """Trimmed time -> original time (what review and actions.csv use)."""
    for m in tmap:
        if m["trim_start_s"] <= t <= m["trim_end_s"]:
            return round(m["orig_start_s"] + t - m["trim_start_s"], 3)
    return None


def _selector(segs: list[dict[str, Any]], fps: float, mode: Mode, lite_fps: float,
              last_frame: int | None = None) -> Callable[[int], int | None]:
    """frame index -> output pts (in frames), or None to drop it. keep_lite also keeps `last_frame`, so the
    file ends where the clip ends."""
    ranges = [(_first_frame(s["start_s"], fps), _first_frame(s["end_s"], fps), s["mode"] == "keep") for s in segs]
    offsets, n = [], 0
    for a, b, keep in ranges:
        offsets.append(n)
        n += max(0, b - a) if keep else 0
    lite_every = max(1, int(round(fps / lite_fps)))

    def pick(i: int) -> int | None:
        for (a, b, keep), off in zip(ranges, offsets, strict=True):
            if a <= i < b:
                if keep:
                    return off + i - a if mode == "cut" else i
                if mode == "cut":
                    return None
                return i if i % lite_every == 0 or i == last_frame else None
        return None

    return pick


def write_trimmed(src: Path, out: Path, segs: list[dict[str, Any]], fps: float, cfg: TrimConfig, mode: Mode = "keep_lite",
                  n_frames: int | None = None, progress: Callable[[float], None] | None = None) -> dict[str, Any]:
    """MP4 of the selected frames. keep_lite: original pts (variable frame rate); cut: consecutive pts."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.partial{out.suffix}")
    pick = _selector(segs, fps, mode, cfg.lite_fps, n_frames - 1 if n_frames else None)
    tb = Fraction(1, int(round(fps)))
    opts = {"crf": str(cfg.crf), "preset": cfg.preset} if cfg.codec.startswith("lib") else {}
    if mode == "keep_lite":
        # B-frames plus gaps in the timestamps make the MP4 report a wrong duration (177.1 s for a 180 s
        # keep-lite clip, measured), so players stopped early. Without B-frames the duration is exact.
        opts["bf"] = "0"
    kept = lite = 0
    keep_spans = [(_first_frame(s["start_s"], fps), _first_frame(s["end_s"], fps)) for s in segs if s["mode"] == "keep"]
    try:
        with av.open(str(tmp), mode="w", format="mp4", options={"movflags": "+faststart"}) as c:
            st = None
            for i, img in iter_all_frames(str(src), cfg.width):
                if progress is not None and n_frames and i % 150 == 0:
                    progress(min(1.0, i / n_frames))
                pts = pick(i)
                if pts is None:
                    continue
                if st is None:
                    st = c.add_stream(cfg.codec, rate=Fraction(fps).limit_denominator(1001))
                    st.width, st.height, st.pix_fmt = img.shape[1], img.shape[0], "yuv420p"
                    st.codec_context.time_base = tb
                    st.options = opts
                fr = av.VideoFrame.from_ndarray(img, format="bgr24")
                fr.pts, fr.time_base = pts, tb
                for p in st.encode(fr):
                    c.mux(p)
                in_keep = any(a <= i < b for a, b in keep_spans)
                kept += in_keep
                lite += not in_keep
            if st is not None:
                for p in st.encode():
                    c.mux(p)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return {"keep_frames": kept, "lite_frames": lite, "lite_every_n": max(1, int(round(fps / cfg.lite_fps)))}


def load_map(out_dir: Path) -> dict[str, Any]:
    p = out_dir / "trim_map.json"
    if not p.is_file():
        return {"runs": {}}
    m = json.loads(p.read_text(encoding="utf-8"))
    if "runs" not in m:  # first version: one keep-lite run, no mode key
        m = {"runs": {"keep_lite": m}}
    return m


def run_trim(out_dir: Path, sanitized: Path, original: Path, actions: list[Action], duration_s: float, fps: float,
             cfg: TrimConfig, mode: Mode | None = None, progress: Callable[[float], None] | None = None) -> dict[str, Any]:
    mode = mode or cfg.mode
    segs = plan(actions, duration_s, cfg)
    dst = cfg.out_dir / f"{out_dir.name}_{mode}.mp4"
    counts = write_trimmed(sanitized, dst, segs, fps, cfg, mode, int(round(duration_s * fps)), progress)
    tmap = time_map(segs, fps)
    info: dict[str, Any] = {
        "mode": mode, "trimmed_path": dst.as_posix(), "source": sanitized.as_posix(), "plan_sha": plan_sha(segs),
        "segments": segs, **summary(segs, duration_s),
        "output_duration_s": round(duration_s if mode == "keep_lite" else (tmap[-1]["trim_end_s"] if tmap else 0.0), 1),
        "size_mb": {"original": round(os.path.getsize(original) / 1e6, 1) if original.is_file() else None,
                    "sanitized": round(os.path.getsize(sanitized) / 1e6, 1),
                    "trimmed": round(os.path.getsize(dst) / 1e6, 1)},
        **counts, "config": cfg.model_dump(mode="json"), "created": time.time(),
    }
    if mode == "cut":
        info["time_map"] = tmap
        info["time_note"] = "trimmed video is shorter; map trimmed time to original time with time_map"
        from egovision.export import write_actions_csv

        moved = []
        for a in kept_actions(actions, cfg):
            s, e = to_trimmed(a.start_s, tmap), to_trimmed(a.end_s, tmap)
            if s is not None and e is not None and e > s:
                moved.append(a.model_copy(update={"start_s": s, "end_s": e}))
        write_actions_csv(moved, out_dir / "actions_trimmed.csv")
    else:
        info["time_note"] = "keep-lite: frames keep their original timestamps, so trimmed t == original t"
    m = load_map(out_dir)
    m["runs"][mode] = info
    m["last"] = mode
    (out_dir / "trim_map.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return info
