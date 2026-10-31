"""End-to-end orchestration: video -> hands -> actions -> files (+ overlay video)."""

from __future__ import annotations

import json
import logging
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from egovision import __version__
from egovision.actions import make_backend as make_actions
from egovision.actions.postprocess import clean
from egovision.config import Config
from egovision.export import write_actions_csv, write_actions_json, write_hands_jsonl
from egovision.hands import make_backend as make_hands
from egovision.hands.postprocess import assign_side_ids, fill_gaps
from egovision.ingest import FrameSource, probe
from egovision.schema import Action, FrameHands, VideoMeta

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    meta: VideoMeta
    out_dir: Path
    hands: list[FrameHands]
    actions: list[Action]
    stats: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def hand_stats(processed: list[FrameHands], filled: list[FrameHands], width: int) -> dict[str, Any]:
    n = max(len(processed), 1)
    sides = [h.side for fh in processed for h in fh.hands]
    # Sanity check for the handedness convention: in egocentric video the wearer's left hand is
    # usually in the left half of the image. A low value suggests swap_handedness is wrong.
    on_expected_half = [
        (h.kp2d[0][0] < width / 2) == (h.side == "left") for fh in processed for h in fh.hands
    ]
    return {
        "processed_frames": len(processed),
        "processed_with_hand_pct": round(100 * sum(bool(fh.hands) for fh in processed) / n, 1),
        "processed_with_two_hands_pct": round(100 * sum(len(fh.hands) >= 2 for fh in processed) / n, 1),
        "all_frames_with_hand_pct": round(100 * sum(bool(fh.hands) for fh in filled) / max(len(filled), 1), 1),
        "left_detections": sides.count("left"),
        "right_detections": sides.count("right"),
        "mean_score": round(float(np.mean([h.score for fh in processed for h in fh.hands])), 3) if sides else None,
        "side_matches_image_half_pct": round(100 * float(np.mean(on_expected_half)), 1) if sides else None,
    }


def run_video(video: str | Path, out_root: str | Path, cfg: Config, render: bool | None = None) -> RunResult:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    t_start = time.perf_counter()
    meta = probe(video, cfg.ingest.min_duration_s)
    src = FrameSource(meta, cfg.ingest)
    n_eff = min(meta.n_frames, src.indices[-1] + 1) if src.indices else 0
    if cfg.ingest.max_seconds is not None:
        n_eff = min(meta.n_frames, int(round(cfg.ingest.max_seconds * meta.fps)))
    meta_eff = meta.model_copy(update={"n_frames": n_eff, "duration_s": round(n_eff / meta.fps, 3)})
    out_dir = Path(out_root) / meta.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("run: %s (%dx%d @ %.2f fps, %.1fs) -> %s; %d frames to process (stride %d, width %d)",
             meta.name, meta.width, meta.height, meta.fps, meta_eff.duration_s, out_dir, len(src),
             cfg.ingest.stride, src.size[0])

    # Stage 1+2: ingest and per-frame hands (timed separately)
    t_ingest = t_hands = 0.0
    processed: list[FrameHands] = []
    with make_hands(cfg.hands, meta) as hb:
        it = iter(src)
        while True:
            t0 = time.perf_counter()
            try:
                fr = next(it)
            except StopIteration:
                break
            t1 = time.perf_counter()
            hands = hb.process(fr)
            t_hands += time.perf_counter() - t1
            t_ingest += t1 - t0
            processed.append(FrameHands(frame=fr.index, t=round(fr.t, 4), hands=hands))
            if len(processed) % 500 == 0:
                log.info("hands: %d/%d frames", len(processed), len(src))

    t0 = time.perf_counter()
    processed = assign_side_ids(processed)
    filled = fill_gaps(processed, n_eff, meta.fps, cfg.hands.max_interp_gap)
    t_post = time.perf_counter() - t0

    # Stage 3: actions
    t0 = time.perf_counter()
    backend = make_actions(cfg.actions)
    actions = clean(backend.segment(meta_eff, filled), meta_eff.duration_s)
    t_actions = time.perf_counter() - t0

    write_hands_jsonl(filled, out_dir / "hands.jsonl")
    write_actions_csv(actions, out_dir / "actions.csv")
    write_actions_json(actions, out_dir / "actions.json")

    t_render = 0.0
    if cfg.render.enabled if render is None else render:
        from egovision.render import render_overlay

        t0 = time.perf_counter()
        enc = render_overlay(meta, filled, actions, cfg.actions.verbs, cfg.render, out_dir / "overlay.mp4")
        t_render = time.perf_counter() - t0
        log.info("render: overlay.mp4 written with encoder %s", enc)

    k = max(len(processed), 1)
    timings = {
        "ingest_s": round(t_ingest, 2), "ingest_fps": round(k / max(t_ingest, 1e-9), 1),
        "hands_s": round(t_hands, 2), "hands_fps": round(k / max(t_hands, 1e-9), 1),
        "postprocess_s": round(t_post, 2), "actions_s": round(t_actions, 2), "render_s": round(t_render, 2),
        "total_s": round(time.perf_counter() - t_start, 2),
    }
    timings["realtime_factor"] = round(meta_eff.duration_s / max(timings["total_s"] - t_render, 1e-9), 2)
    stats = hand_stats(processed, filled, meta.width)
    stats["actions"] = len(actions)
    stats["actions_backend"] = backend.name
    run_info = {
        "egovision_version": __version__, "video": meta.model_dump(), "processed_until_s": meta_eff.duration_s,
        "config": json.loads(cfg.model_dump_json()), "timings": timings, "stats": stats,
        "host": {"python": platform.python_version(), "machine": platform.processor() or platform.machine()},
    }
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=1), encoding="utf-8")
    log.info("done: %s timings=%s stats=%s", meta.name, timings, stats)
    return RunResult(meta=meta_eff, out_dir=out_dir, hands=filled, actions=actions, stats=stats, timings=timings)
