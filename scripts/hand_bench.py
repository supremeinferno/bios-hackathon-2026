"""Hand-detection variants on real clips, without ground truth: raw detections, label-free diagnostics and
a differential audit sheet (hands one variant finds that the baseline does not, and vice versa).

    python scripts/hand_bench.py --clips mytronlabs-000060-016 MTL-SKL00089-007 --seconds 60 \
        --variants baseline conf03 tiles3 --out outputs_exp/hands

Why differential: with no labelled hands, "more detections" can be more false hands (pink rollers, feet,
coworkers; docs/ERRORS_AND_FIXES.md #30). Only the hands a change ADDS or REMOVES need checking, so the
script tiles exactly those (artifacts/hand_audit/, git-ignored) and writes an index CSV to judge them.

Variants (MediaPipe Hand Landmarker; frames from the 960 px stride-2 cache unless noted):
  baseline     VIDEO mode, detection/presence/tracking confidence 0.4 (current default)
  conf03       same, thresholds 0.3
  conf05       same, thresholds 0.5
  fullres      VIDEO mode on 1920 px frames (the palm detector input is 192 px, so this mainly tests JPEG/
               downscale effects)
  tiles3       baseline + IMAGE mode on three overlapping square tiles across the frame; a hand is
               1.8x larger in the 192 px palm-detector input. Merged with box NMS, highest score kept
  tiles3_miss  tiles only on frames where the baseline found fewer than 2 hands
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from egovision.config import Config
from egovision.hands.mediapipe_backend import MediaPipeHands
from egovision.hands.tracking import smooth_tracks, track_hands
from egovision.ingest import FrameSource, probe
from egovision.metrics import box_iou, hand_diagnostics
from egovision.schema import FrameHands, Hand

VARIANTS = ["baseline", "conf03", "conf05", "fullres", "tiles3", "tiles3_miss"]


def tiles(w: int, h: int, n: int = 3) -> list[tuple[int, int, int, int]]:
    side = min(w, h)
    xs = np.linspace(0, w - side, n).round().astype(int)
    return [(int(x), 0, int(x) + side, side) for x in xs]


def nms(hands: list[Hand], iou: float = 0.3) -> list[Hand]:
    kept: list[Hand] = []
    for h in sorted(hands, key=lambda h: -h.score):
        if all(box_iou(h.bbox, k.bbox) < iou for k in kept):
            kept.append(h)
    return kept


class TileDetector:
    """IMAGE-mode landmarker on sub-images; landmarks mapped back to original-video pixels."""

    def __init__(self, base: MediaPipeHands):
        from mediapipe.tasks.python import vision

        self.base = base
        self.lm = vision.HandLandmarker.create_from_options(base._opts(vision.RunningMode.IMAGE))

    def __call__(self, img: np.ndarray, scale: float) -> list[Hand]:
        h, w = img.shape[:2]
        out = []
        for x1, y1, x2, y2 in tiles(w, h):
            rgb = cv2.cvtColor(np.ascontiguousarray(img[y1:y2, x1:x2]), cv2.COLOR_BGR2RGB)
            res = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            tw, th = (x2 - x1) / scale, (y2 - y1) / scale
            ox, oy = x1 / scale, y1 / scale
            for pts, handed in zip(res.hand_landmarks, res.handedness, strict=True):
                kp = [[ox + min(max(p.x, 0.0), 1.0) * tw, oy + min(max(p.y, 0.0), 1.0) * th] for p in pts]
                xs, ys = [k[0] for k in kp], [k[1] for k in kp]
                out.append(Hand(side=handed[0].category_name.lower(), score=float(handed[0].score),  # type: ignore[arg-type]
                                source="recovered", bbox=[min(xs), min(ys), max(xs), max(ys)], kp2d=kp))
        return out

    def close(self) -> None:
        self.lm.close()


def run_variant(clip: str, variant: str, seconds: float) -> tuple[list[FrameHands], dict]:
    over = [f"ingest.max_seconds={seconds}"]
    if variant == "conf03":
        over += [f"hands.min_{k}_confidence=0.3" for k in ("detection", "presence", "tracking")]
    if variant == "conf05":
        over += [f"hands.min_{k}_confidence=0.5" for k in ("detection", "presence", "tracking")]
    if variant == "fullres":
        over += ["ingest.width=0", "ingest.cache_frames=false"]
    cfg = Config.load(None, over)
    meta = probe(Path("data/raw") / f"{clip}.mp4")
    src = FrameSource(meta, cfg.ingest)
    frames: list[FrameHands] = []
    t_model = 0.0
    with MediaPipeHands(cfg.hands, meta) as hb:
        tile = TileDetector(hb) if variant.startswith("tiles3") else None
        for fr in src:
            t0 = time.perf_counter()
            hands = hb.process(fr)
            if tile is not None and (variant == "tiles3" or len(hands) < 2):
                hands = nms(hands + tile(fr.image, fr.scale))
            t_model += time.perf_counter() - t0
            frames.append(FrameHands(frame=fr.index, t=round(fr.t, 4), hands=hands))
        if tile is not None:
            tile.close()
    return frames, {"frames": len(frames), "model_fps": round(len(frames) / max(t_model, 1e-9), 1),
                    "fps": meta.fps, "width": meta.width, "height": meta.height}


def diff_hands(a: list[FrameHands], b: list[FrameHands], iou: float = 0.3) -> list[tuple[int, Hand]]:
    """Hands in `a` with no hand in `b` (same frame) overlapping by IoU >= iou."""
    bf = {fh.frame: fh.hands for fh in b}
    return [(fh.frame, h) for fh in a for h in fh.hands if all(box_iou(h.bbox, g.bbox) < iou for g in bf.get(fh.frame, []))]


def audit_sheet(clip: str, items: list[tuple[int, Hand]], tag: str, out_dir: Path, n: int,
                rng: np.random.Generator) -> list[dict]:
    if not items:
        return []
    pick = sorted(rng.choice(len(items), size=min(n, len(items)), replace=False).tolist())
    cfg = Config.load(None, [])
    meta = probe(Path("data/raw") / f"{clip}.mp4")
    src = FrameSource(meta, cfg.ingest)
    scale = src.scale
    want = sorted({items[i][0] for i in pick})
    imgs = {f.index: f.image for f in src.load(want)}
    tiles_out, rows = [], []
    for k, i in enumerate(pick):
        frame, h = items[i]
        img = imgs[frame].copy()
        x1, y1, x2, y2 = (int(v * scale) for v in h.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for x, y in h.kp2d:
            cv2.circle(img, (int(x * scale), int(y * scale)), 2, (0, 0, 255), -1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = max(x2 - x1, y2 - y1, 60)
        H, W = img.shape[:2]
        crop = img[max(0, cy - half):min(H, cy + half), max(0, cx - half):min(W, cx + half)]
        crop = cv2.resize(crop, (220, 220))
        ctx = cv2.resize(img, (391, 220))
        tile = np.hstack([crop, ctx])
        cv2.putText(tile, f"{tag}#{k} f{frame} {h.side[0].upper()} {h.score:.2f}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        tiles_out.append(tile)
        rows.append({"clip": clip, "set": tag, "idx": k, "frame": frame, "side": h.side, "score": round(h.score, 3),
                     "bbox": [round(v, 1) for v in h.bbox], "judgement": "", "judge": ""})
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(tiles_out), 12):
        grid = tiles_out[s:s + 12]
        while len(grid) % 2:
            grid.append(np.zeros_like(grid[0]))
        sheet = np.vstack([np.hstack(grid[j:j + 2]) for j in range(0, len(grid), 2)])
        cv2.imwrite(str(out_dir / f"{clip}_{tag}_{s // 12:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--variants", nargs="+", default=["baseline", "tiles3"], choices=VARIANTS)
    ap.add_argument("--out", type=Path, default=Path("outputs_exp/hands"))
    ap.add_argument("--audit", type=Path, default=Path("artifacts/hand_audit"))
    ap.add_argument("--audit-n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cfg = Config().hands
    report: dict = {}
    audit_rows: list[dict] = []
    for clip in a.clips:
        report[clip] = {}
        runs: dict[str, list[FrameHands]] = {}
        for v in a.variants:
            raw_path = a.out / v / f"{clip}_{int(a.seconds)}s.raw.jsonl"
            info_path = raw_path.with_suffix(".info.json")
            if raw_path.is_file() and info_path.is_file():
                frames = [FrameHands.model_validate_json(x) for x in raw_path.read_text(encoding="utf-8").splitlines()]
                info = json.loads(info_path.read_text(encoding="utf-8"))
            else:
                frames, info = run_variant(clip, v, a.seconds)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text("\n".join(fh.model_dump_json() for fh in frames), encoding="utf-8")
                info_path.write_text(json.dumps(info), encoding="utf-8")
            runs[v] = frames
            tracked = smooth_tracks(track_hands(frames, info["fps"], info["width"], info["height"], cfg), cfg)
            d = hand_diagnostics(tracked, info["fps"], info["width"], stride=2)
            d["model_fps"] = info["model_fps"]
            report[clip][v] = d
            print(clip, v, json.dumps(d), flush=True)
        if "baseline" in runs:
            for v, frames in runs.items():
                if v == "baseline":
                    continue
                added, lost = diff_hands(frames, runs["baseline"]), diff_hands(runs["baseline"], frames)
                report[clip][v]["added_vs_baseline"] = len(added)
                report[clip][v]["lost_vs_baseline"] = len(lost)
                audit_rows += audit_sheet(clip, added, f"{v}+", a.audit, a.audit_n, rng)
                audit_rows += audit_sheet(clip, lost, f"{v}-", a.audit, a.audit_n // 2, rng)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"bench_{int(a.seconds)}s.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if audit_rows:
        p = a.audit / f"audit_index_{int(a.seconds)}s.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0]))
            w.writeheader()
            w.writerows(audit_rows)
        print(f"audit sheets in {a.audit}/, index {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
