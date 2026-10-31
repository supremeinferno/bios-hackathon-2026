"""End-to-end orchestration: video -> hands -> objects/contacts -> steps -> files (+ overlay video)."""

from __future__ import annotations

import json
import logging
import platform
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from egovision import __version__
from egovision.analytics import clip_analytics
from egovision.config import Config
from egovision.device import resolve_device
from egovision.export import write_outputs, write_signals_csv
from egovision.hands import make_backend as make_hands
from egovision.hands.postprocess import fill_gaps
from egovision.hands.tracking import smooth_tracks, track_hands
from egovision.ingest import FrameSource, probe
from egovision.profile import load_profile
from egovision.qc import frame_qc
from egovision.qc import summarize as summarize_qc
from egovision.schema import Action, ContactSegment, FrameHands, FrameObjects, VideoMeta
from egovision.signals import build_signals
from egovision.steps import make_backend as make_steps
from egovision.steps.features import SIDES, KeyframeFeatures, SideFeat, crop_box, cut, hand_speed, main_hand
from egovision.steps.postprocess import clean
from egovision.verify import status_counts, verify_and_refine

log = logging.getLogger(__name__)
Progress = Callable[[str, float], None]  # (stage, fraction 0..1)


@dataclass
class RunResult:
    meta: VideoMeta
    out_dir: Path
    hands: list[FrameHands]
    objects: list[FrameObjects]
    contacts: list[ContactSegment]
    actions: list[Action]
    stats: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def hand_stats(processed: list[FrameHands], filled: list[FrameHands], width: int) -> dict[str, Any]:
    n = max(len(processed), 1)
    sides = [h.side for fh in processed for h in fh.hands]
    # In egocentric video the wearer's left hand is usually in the left half of the image; a low
    # value means handedness labels are probably reversed (see docs/DECISIONS.md D4).
    on_half = [(h.kp2d[0][0] < width / 2) == (h.side == "left") for fh in processed for h in fh.hands]
    ids = {h.id for fh in processed for h in fh.hands if h.id is not None}
    return {
        "processed_frames": len(processed),
        "processed_with_hand_pct": round(100 * sum(bool(fh.hands) for fh in processed) / n, 1),
        "processed_with_two_hands_pct": round(100 * sum(len(fh.hands) >= 2 for fh in processed) / n, 1),
        "all_frames_with_hand_pct": round(100 * sum(bool(fh.hands) for fh in filled) / max(len(filled), 1), 1),
        "left_detections": sides.count("left"),
        "right_detections": sides.count("right"),
        "track_ids": len(ids),
        "mean_score": round(float(np.mean([h.score for fh in processed for h in fh.hands])), 3) if sides else None,
        "side_matches_image_half_pct": round(100 * float(np.mean(on_half)), 1) if sides else None,
    }


def keyframe_indices(src: FrameSource, every_s: float) -> list[int]:
    every = max(1, int(round(every_s * src.meta.fps / src.cfg.stride)))
    return src.indices[::every]


def run_video(video: str | Path, out_root: str | Path, cfg: Config, render: bool | None = None,
              progress: Progress | None = None) -> RunResult:
    def report(stage: str, frac: float) -> None:
        if progress is not None:
            progress(stage, frac)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    t_start = time.perf_counter()
    device = resolve_device(cfg.device)
    meta_full = probe(video, cfg.ingest.min_duration_s)
    src = FrameSource(meta_full, cfg.ingest)
    n_eff = meta_full.n_frames
    if cfg.ingest.max_seconds is not None:
        n_eff = min(n_eff, int(round(cfg.ingest.max_seconds * meta_full.fps)))
    meta = meta_full.model_copy(update={"n_frames": n_eff, "duration_s": round(n_eff / meta_full.fps, 3)})
    out_dir = Path(out_root) / meta.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("run: %s (%dx%d @ %.2f fps, %.1fs) -> %s | device=%s | %d frames (stride %d, width %d)",
             meta.name, meta.width, meta.height, meta.fps, meta.duration_s, out_dir, device, len(src),
             cfg.ingest.stride, src.size[0])
    timings: dict[str, float] = {}

    # ---- Stage: ingest + hands -------------------------------------------------------------
    t_ingest = t_hands = t_qc = 0.0
    processed: list[FrameHands] = []
    qc_raw: list[tuple[float, float, float]] = []
    profile = load_profile(cfg.profile_path, meta.name)
    log.info("profile: %s expects %s%s", profile.source, profile.expected, " (gloved)" if profile.any_gloved else "")
    with make_hands(cfg.hands, meta_full, expected=len(profile.expected), gloved=profile.any_gloved) as hb:
        it = iter(src)
        while True:
            t0 = time.perf_counter()
            try:
                fr = next(it)
            except StopIteration:
                break
            t1 = time.perf_counter()
            processed.append(FrameHands(frame=fr.index, t=round(fr.t, 4), hands=hb.process(fr)))
            t2 = time.perf_counter()
            t_hands += t2 - t1
            if cfg.qc.enabled:
                qc_raw.append(frame_qc(fr.image, cfg.qc.size))
                t_qc += time.perf_counter() - t2
            t_ingest += t1 - t0
            if len(processed) % 250 == 0:
                log.info("hands: %d/%d frames", len(processed), len(src))
                report("hands", len(processed) / max(len(src), 1))
        hand_counters = dict(getattr(hb, "counters", {}))
    k = max(len(processed), 1)
    timings.update(ingest_s=round(t_ingest, 2), ingest_fps=round(k / max(t_ingest, 1e-9), 1),
                   hands_s=round(t_hands, 2), hands_fps=round(k / max(t_hands, 1e-9), 1), qc_s=round(t_qc, 2))
    qc = summarize_qc([fh.frame for fh in processed], meta.fps, qc_raw, cfg.qc) if cfg.qc.enabled else None
    if qc is not None:
        (out_dir / "qc.json").write_text(json.dumps(qc), encoding="utf-8")
        log.info("qc: %s", qc["summary"])

    # Raw per-frame detections before tracking: lets tracking / verification experiments replay without
    # re-running the hand model (scripts/replay.py).
    with open(out_dir / "hands_raw.jsonl", "w", encoding="utf-8") as f:
        for fh in processed:
            f.write(fh.model_dump_json() + "\n")
    t0 = time.perf_counter()
    tracked = track_hands(processed, meta.fps, meta.width, meta.height, cfg.hands)
    if cfg.hands.smooth:
        tracked = smooth_tracks(tracked, cfg.hands)
    filled = fill_gaps(tracked, n_eff, meta.fps, cfg.hands.max_interp_gap)
    timings["tracking_s"] = round(time.perf_counter() - t0, 2)
    log.info("tracking: %d track ids", len({h.id for fh in tracked for h in fh.hands}))
    sig_rows: list[dict[str, Any]] = []
    if cfg.signals.enabled or cfg.verify.enabled:
        sig_rows = build_signals(filled, meta.fps, meta.width, meta.height, cfg.signals,
                                 qc["frames"] if qc else None)
        write_signals_csv(sig_rows, out_dir / "signals.csv")

    # ---- Stage: objects + privacy detections + keyframe features (one batched pass) --------
    t0 = time.perf_counter()
    # The offline matcher always runs: it computes the keyframe embeddings, it is the fallback when the
    # VLM fails, and its output is kept for hand-grounded verification of VLM actions.
    steps_backend = make_steps(cfg.steps.model_copy(update={"backend": "exemplar"}), device)
    vlm = make_steps(cfg.steps, device) if cfg.steps.backend == "gemini" else None
    key_idx = keyframe_indices(src, cfg.objects.every_s)
    detector = None
    vocab = (cfg.objects.vocab_for(meta.name) if cfg.objects.enabled else []) + \
        (list(cfg.privacy.prompts) if cfg.privacy.enabled else [])
    if vocab:
        from egovision.objects import OpenVocabDetector

        detector = OpenVocabDetector(cfg.objects, device)
    sanitizer = None
    if cfg.privacy.enabled:
        from egovision.privacy import Sanitizer

        sanitizer = Sanitizer(meta_full, cfg.privacy)
    log.info("keyframes: %d, %d prompts (%s)", len(key_idx), len(vocab), ", ".join(v.name for v in vocab))
    from egovision.objects import associate, contact_segments, nearest_object

    lag = max(1, int(round(cfg.steps.speed_lag_s * meta.fps)))
    frame_objects: list[FrameObjects] = []
    keyframes: list[KeyframeFeatures] = []
    t_det = t_emb = 0.0
    batch: list = []

    def flush() -> None:
        nonlocal t_det, t_emb
        if not batch:
            return
        td = time.perf_counter()
        dets = detector.detect([f.image for f in batch], vocab, 1 / src.scale) if detector else [[] for _ in batch]
        t_det += time.perf_counter() - td
        crops, owners = [], []
        for f, all_dets in zip(batch, dets, strict=True):
            fh = filled[f.index]
            objs = [d for d in all_dets if d.kind != "privacy"]
            if sanitizer is not None:
                sanitizer.add(f.index, round(f.t, 3), [d for d in all_dets if d.kind == "privacy"], fh.hands)
            contacts = associate(fh.hands, objs, cfg.objects)
            frame_objects.append(FrameObjects(frame=f.index, t=round(f.t, 3), objects=objs, contacts=contacts))
            kf = KeyframeFeatures(frame=f.index, t=round(f.t, 3), objects=objs)
            for side in SIDES:
                h = main_hand(fh, side)
                if h is None:
                    kf.sides[side] = SideFeat(present=False)
                    continue
                con = next((c for c in contacts if c.hand_id == h.id), None)
                box = crop_box(h, con, cfg.steps.crop_pad, cfg.steps.crop_min_px, meta.width, meta.height)
                near = nearest_object(h, objs, cfg.steps.object_fallback_max_dist)
                kf.sides[side] = SideFeat(present=True, hand_id=h.id, contact=con, crop_box=box,
                                          speed=hand_speed(filled, f.index, side, lag, meta.fps),
                                          near_object=near.label if near else None)
                if steps_backend.needs_embeddings:
                    crops.append(cut(f.image, box, src.scale))
                    owners.append((kf, side))
            keyframes.append(kf)
        if crops:
            te = time.perf_counter()
            embs = steps_backend.embedder.images(crops)  # type: ignore[attr-defined]
            t_emb += time.perf_counter() - te
            for (kf, side), e in zip(owners, embs, strict=True):
                kf.sides[side].emb = e
        batch.clear()

    for n, f in enumerate(src.load(key_idx)):
        batch.append(f)
        if len(batch) >= max(cfg.objects.batch_size, 8):
            flush()
            report("objects", (n + 1) / max(len(key_idx), 1))
    flush()
    step_s = cfg.objects.every_s
    contacts = contact_segments(frame_objects, meta.name, step_s, cfg.objects.contact_min_s)
    nk = max(len(key_idx), 1)
    timings.update(objects_s=round(t_det, 2), objects_fps=round(nk / max(t_det, 1e-9), 1),
                   embed_s=round(t_emb, 2), keyframe_pass_s=round(time.perf_counter() - t0, 2))

    # ---- Stage 1: privacy sanitization (sanitized copy for API, UI, demos) -----------------
    sanitized: Path | None = None
    privacy_summary: dict[str, Any] = {"enabled": False}
    if sanitizer is not None:
        t0 = time.perf_counter()
        report("privacy", 0.0)
        sanitized = cfg.privacy.out_dir / f"{meta.name}.mp4"
        privacy_summary = sanitizer.write(filled, sanitized)
        (out_dir / "privacy_log.json").write_text(json.dumps(sanitizer.log_record(privacy_summary), indent=1),
                                                  encoding="utf-8")
        timings["privacy_s"] = round(time.perf_counter() - t0, 2)
        log.info("privacy: %s", {k: v for k, v in privacy_summary.items() if k != "note"})

    # ---- Stage: steps ---------------------------------------------------------------------
    t0 = time.perf_counter()
    report("steps", 0.0)
    offline_actions = clean(steps_backend.segment(meta, filled, keyframes, contacts), meta.duration_s)
    actions, used, vlm_actions, vlm_error = offline_actions, steps_backend.name, None, None
    vlm_models: dict[str, int] = {}
    if vlm is not None:
        vlm.source_video = sanitized  # type: ignore[attr-defined]  # only sanitized video leaves the machine
        try:
            vlm_actions = clean(vlm.segment(meta, filled, keyframes, contacts), meta.duration_s)
            actions, used = vlm_actions, f"{vlm.name}:{getattr(vlm, 'model_used', '')}"
            vlm_models = dict(getattr(vlm, "models_used", {}))
        except Exception as e:  # noqa: BLE001 - a VLM outage must not lose the rest of the run
            vlm_error = f"{type(e).__name__}: {e}"[:500]
            log.error("steps: %s failed, using the offline timeline instead: %s", vlm.name, vlm_error)
    (out_dir / "actions_offline.json").write_text(
        json.dumps([a.model_dump() for a in offline_actions], indent=1), encoding="utf-8")
    if vlm_actions is not None:
        (out_dir / "actions_vlm.json").write_text(
            json.dumps([a.model_dump() for a in vlm_actions], indent=1), encoding="utf-8")
    timings["steps_s"] = round(time.perf_counter() - t0, 2)

    # ---- Stages 7-8: hand-grounded verification + boundary snapping ----------------------
    if cfg.verify.enabled and sig_rows:
        t0 = time.perf_counter()
        actions = clean(verify_and_refine(actions, sig_rows, contacts, cfg.verify, profile), meta.duration_s)
        timings["verify_s"] = round(time.perf_counter() - t0, 2)
        log.info("verify: %s", status_counts(actions))
    actions = [a.model_copy(update={"id": f"a{i:03d}"}) for i, a in enumerate(actions)]  # stable ids for review
    (out_dir / "analytics.json").write_text(json.dumps(clip_analytics(actions, sig_rows, meta.duration_s), indent=1),
                                            encoding="utf-8")

    stats = hand_stats(tracked, filled, meta.width)
    stats["profile"] = {"source": profile.source, "expected": profile.expected, "gloved": profile.any_gloved}
    stats["expected_hand_presence_pct"] = {
        s: round(100 * sum(any(h.side == s for h in fh.hands) for fh in tracked) / max(len(tracked), 1), 1)
        for s in profile.expected}
    stats["hand_sources"] = {src: sum(h.source == src for fh in filled for h in fh.hands)
                             for src in ("detected", "recovered", "interpolated")}
    stats.update(hand_counters)
    stats.update(keyframes=len(keyframes), objects_detected=sum(len(fo.objects) for fo in frame_objects),
                 keyframes_with_contact_pct=round(100 * sum(any(c.state != "none" for c in fo.contacts)
                                                            for fo in frame_objects) / nk, 1),
                 contact_segments=len(contacts), actions=len(actions), steps_backend=used,
                 steps_requested=cfg.steps.backend, vlm_error=vlm_error, vlm_windows_by_model=vlm_models,
                 verification=status_counts(actions),
                 device=device, privacy=privacy_summary)

    # ---- Export -----------------------------------------------------------------------------
    t0 = time.perf_counter()
    write_outputs(out_dir, meta, filled, frame_objects, contacts, actions, keyframes, sanitized,
                  drop_rejected=cfg.verify.enabled and cfg.verify.drop_rejected)
    timings["export_s"] = round(time.perf_counter() - t0, 2)
    t_render = 0.0
    if cfg.render.enabled if render is None else render:
        from egovision.render import render_overlay

        report("render", 0.0)
        t0 = time.perf_counter()
        enc = render_overlay(meta, filled, frame_objects, actions, cfg.steps.verbs, cfg.render,
                             out_dir / "overlay.mp4", source=sanitized)
        t_render = time.perf_counter() - t0
        log.info("render: overlay.mp4 written with encoder %s", enc)
    timings["render_s"] = round(t_render, 2)
    timings["total_s"] = round(time.perf_counter() - t_start, 2)
    timings["realtime_factor"] = round(meta.duration_s / max(timings["total_s"] - t_render, 1e-9), 2)
    run_info = {
        "egovision_version": __version__, "video": meta_full.model_dump(), "processed_until_s": meta.duration_s,
        "config": json.loads(cfg.model_dump_json()), "timings": timings, "stats": stats,
        "host": {"python": platform.python_version(), "machine": platform.processor() or platform.machine()},
    }
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=1), encoding="utf-8")
    report("done", 1.0)
    log.info("done: %s timings=%s stats=%s", meta.name, timings, stats)
    return RunResult(meta=meta, out_dir=out_dir, hands=filled, objects=frame_objects, contacts=contacts,
                     actions=actions, stats=stats, timings=timings)
