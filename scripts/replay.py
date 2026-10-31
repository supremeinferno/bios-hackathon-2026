"""Re-run every stage after hand detection from a finished run, with a different config, in seconds.

    python scripts/replay.py --from outputs_exp/E0_baseline --to outputs_exp/E2_tracking \
        --config experiments/configs/E2_tracking.yaml [--raw-hands outputs_exp/hands/baseline]

Re-used from the source run (expensive, unchanged by the stages replayed):
  raw per-frame hand detections  <raw-hands>/<clip>_180s.raw.jsonl (scripts/hand_bench.py) or
                                 <from>/<clip>/hands_raw.jsonl (written by the pipeline since this audit)
  keyframe object boxes          <from>/<clip>/result.json (OWLv2)
  QC per frame                   <from>/<clip>/qc.json
  offline timeline               <from>/<clip>/actions_offline.json
  Gemini answers                 cache/api (a changed prompt or window plan WILL call the API; it uploads only
                                 the sanitized copy in <from>/_sanitized)
Replayed with the new config: tracking, smoothing, gap filling, signals, contacts, Gemini post-processing,
verification, boundary snapping, export. A replay of the source config must reproduce the source outputs
(checked by --check).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from egovision.config import Config
from egovision.export import deliverable, read_actions_json, write_actions_csv, write_actions_json, write_contacts_csv, write_signals_csv
from egovision.hands.postprocess import fill_gaps
from egovision.hands.tracking import smooth_tracks, track_hands
from egovision.objects import associate, contact_segments
from egovision.profile import load_profile
from egovision.schema import FrameHands, FrameObjects, ObjectDet, VideoMeta
from egovision.signals import build_signals
from egovision.steps.postprocess import clean
from egovision.verify import status_counts, verify_and_refine

log = logging.getLogger("replay")


def load_raw(clip: str, src_run: Path, raw_dir: Path | None, n_frames: int, fps: float) -> list[FrameHands]:
    cands = [src_run / "hands_raw.jsonl"]
    if raw_dir is not None:
        cands.insert(0, raw_dir / f"{clip}_{int(round(n_frames / fps))}s.raw.jsonl")
    for p in cands:
        if p.is_file():
            return [FrameHands.model_validate_json(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    raise FileNotFoundError(f"no raw hand detections for {clip} in {cands}")


def replay_clip(src_run: Path, dst_root: Path, cfg: Config, raw_dir: Path | None) -> dict:
    run = json.loads((src_run / "run.json").read_text(encoding="utf-8"))
    res = json.loads((src_run / "result.json").read_text(encoding="utf-8"))
    meta_full = VideoMeta.model_validate(run["video"])
    dur = float(run["processed_until_s"])
    n_eff = int(round(dur * meta_full.fps))
    meta = meta_full.model_copy(update={"n_frames": n_eff, "duration_s": round(n_eff / meta_full.fps, 3)})
    out = dst_root / meta.name
    out.mkdir(parents=True, exist_ok=True)
    profile = load_profile(cfg.profile_path, meta.name)

    processed = load_raw(meta.name, src_run, raw_dir, n_eff, meta.fps)
    tracked = track_hands(processed, meta.fps, meta.width, meta.height, cfg.hands)
    if cfg.hands.smooth:
        tracked = smooth_tracks(tracked, cfg.hands)
    filled = fill_gaps(tracked, n_eff, meta.fps, cfg.hands.max_interp_gap)
    qc = json.loads((src_run / "qc.json").read_text(encoding="utf-8")) if (src_run / "qc.json").is_file() else None
    sig = build_signals(filled, meta.fps, meta.width, meta.height, cfg.signals, qc["frames"] if qc else None)
    write_signals_csv(sig, out / "signals.csv")

    frame_objects = []
    for k in res["keyframes"]:
        objs = [ObjectDet.model_validate(o) for o in k["objects"]]
        frame_objects.append(FrameObjects(frame=k["frame"], t=k["t"], objects=objs,
                                          contacts=associate(filled[k["frame"]].hands, objs, cfg.objects)))
    contacts = contact_segments(frame_objects, meta.name, cfg.objects.every_s, cfg.objects.contact_min_s)
    write_contacts_csv(contacts, out / "contacts.csv")

    offline = read_actions_json(src_run / "actions_offline.json")
    actions, used = offline, "exemplar"
    if cfg.steps.backend == "gemini":
        from egovision.steps import make_backend

        vlm = make_backend(cfg.steps, "cpu")
        san = src_run.parent / "_sanitized" / f"{meta.name}.mp4"
        vlm.source_video = san if san.is_file() else None  # type: ignore[attr-defined]
        actions = clean(vlm.segment(meta, filled, [], contacts), meta.duration_s)
        used = f"gemini:{getattr(vlm, 'models_used', {})}"
        (out / "actions_vlm.json").write_text(json.dumps([a.model_dump() for a in actions], indent=1), encoding="utf-8")
    if cfg.verify.enabled and sig:
        actions = clean(verify_and_refine(actions, sig, contacts, cfg.verify, profile), meta.duration_s)
    actions = [a.model_copy(update={"id": f"a{i:03d}"}) for i, a in enumerate(actions)]
    write_actions_json(actions, out / "actions.json")
    deliver = deliverable(actions, cfg.verify.enabled and cfg.verify.drop_rejected)
    write_actions_csv(deliver, out / "actions.csv")
    with open(out / "hands.jsonl", "w", encoding="utf-8") as f:
        for fh in filled:
            f.write(fh.model_dump_json() + "\n")
    info = {"replayed_from": str(src_run), "video": meta_full.model_dump(), "processed_until_s": dur,
            "config": json.loads(cfg.model_dump_json()), "timings": {}, "stats": {
                "steps_backend": used, "verification": status_counts(actions), "actions": len(actions),
                "delivered_actions": len(deliver)}}
    (out / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info["stats"]


def same_actions(a: Path, b: Path) -> bool:
    return a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path, required=True)
    ap.add_argument("--to", dest="dst", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--raw-hands", type=Path, default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--check", action="store_true", help="Fail unless actions.csv equals the source run's")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "urllib3", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass
    cfg = Config.load(a.config, a.overrides)
    a.dst.mkdir(parents=True, exist_ok=True)
    (a.dst / "config_used.json").write_text(cfg.model_dump_json(indent=1), encoding="utf-8")
    ok = True
    for run in sorted(p for p in a.src.iterdir() if (p / "result.json").is_file()):
        if a.clips and run.name not in a.clips:
            continue
        stats = replay_clip(run, a.dst, cfg, a.raw_hands)
        print(run.name, json.dumps(stats), flush=True)
        if a.check:
            same = same_actions(run / "actions.csv", a.dst / run.name / "actions.csv")
            print(f"  reproduces source actions.csv: {same}")
            ok &= same
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
