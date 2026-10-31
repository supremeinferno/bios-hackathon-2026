"""Stage 10 smooth trim + compress (FORGE). Runs only after a reviewer approved the trim plan.

Keep-lite: footage outside the kept segments is not cut but kept at `lite_fps`, so the video stays
continuous and every timestamp in actions.csv stays valid (frames keep their original timestamps).
Kept segments = every non-rejected action padded by `pad_s`, merged across gaps < `merge_gap_s`;
no cut ever falls inside an action. Input is the SANITIZED copy; output is H.265 at `width`.
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import av

from egovision.config import TrimConfig
from egovision.ingest import iter_all_frames
from egovision.schema import Action


def plan(actions: list[Action], duration_s: float, cfg: TrimConfig) -> list[dict[str, Any]]:
    spans = sorted((max(0.0, a.start_s - cfg.pad_s), min(duration_s, a.end_s + cfg.pad_s)) for a in actions
                   if not (a.review and a.review.get("status") == "rejected") and a.status != "rejected")
    keep: list[list[float]] = []
    for s, e in spans:
        if keep and s - keep[-1][1] < cfg.merge_gap_s:
            keep[-1][1] = max(keep[-1][1], e)
        else:
            keep.append([s, e])
    segs: list[dict[str, Any]] = []
    t = 0.0
    for s, e in keep:
        if s > t:
            segs.append({"start_s": round(t, 2), "end_s": round(s, 2), "mode": "lite"})
        segs.append({"start_s": round(s, 2), "end_s": round(e, 2), "mode": "keep"})
        t = e
    if t < duration_s:
        segs.append({"start_s": round(t, 2), "end_s": round(duration_s, 2), "mode": "lite"})
    return segs


def write_trimmed(src: Path, out: Path, segs: list[dict[str, Any]], fps: float, cfg: TrimConfig) -> dict[str, Any]:
    """Variable-frame-rate MP4: all frames in 'keep' segments, `lite_fps` frames elsewhere, original pts."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.partial{out.suffix}")
    lite_every = max(1, int(round(fps / cfg.lite_fps)))
    keep_frames = 0
    lite_frames = 0
    try:
        with av.open(str(tmp), mode="w", format="mp4") as c:
            st = None
            for i, img in iter_all_frames(str(src), cfg.width):
                t = i / fps
                seg = next((s for s in segs if s["start_s"] <= t < s["end_s"]), segs[-1] if segs else None)
                lite = seg is not None and seg["mode"] == "lite"
                if lite and i % lite_every:
                    continue
                if st is None:
                    st = c.add_stream(cfg.codec, rate=Fraction(fps).limit_denominator(1001))
                    st.width, st.height, st.pix_fmt = img.shape[1], img.shape[0], "yuv420p"
                    st.codec_context.time_base = Fraction(1, int(round(fps)))
                    st.options = {"crf": str(cfg.crf)} if cfg.codec.startswith("lib") else {}
                fr = av.VideoFrame.from_ndarray(img, format="bgr24")
                fr.pts = i  # original timestamp: timestamps in actions.csv stay valid
                fr.time_base = Fraction(1, int(round(fps)))
                for p in st.encode(fr):
                    c.mux(p)
                lite_frames += lite
                keep_frames += not lite
            if st is not None:
                for p in st.encode():
                    c.mux(p)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return {"keep_frames": keep_frames, "lite_frames": lite_frames, "lite_every_n": lite_every}


def run_trim(out_dir: Path, sanitized: Path, original: Path, actions: list[Action], duration_s: float, fps: float,
             cfg: TrimConfig) -> dict[str, Any]:
    segs = plan(actions, duration_s, cfg)
    dst = cfg.out_dir / f"{out_dir.name}.mp4"
    counts = write_trimmed(sanitized, dst, segs, fps, cfg)
    kept_s = sum(s["end_s"] - s["start_s"] for s in segs if s["mode"] == "keep")
    info = {
        "trimmed_path": dst.as_posix(), "source": sanitized.as_posix(), "segments": segs,
        "kept_full_rate_s": round(kept_s, 1), "kept_full_rate_pct": round(100 * kept_s / max(duration_s, 1e-6), 1),
        "size_mb": {"original": round(os.path.getsize(original) / 1e6, 1),
                    "sanitized": round(os.path.getsize(sanitized) / 1e6, 1),
                    "trimmed": round(os.path.getsize(dst) / 1e6, 1)},
        **counts, "config": cfg.model_dump(mode="json"),
        "time_map": "frames keep their original timestamps (keep-lite), so trimmed t == original t",
    }
    (out_dir / "trim_map.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info
