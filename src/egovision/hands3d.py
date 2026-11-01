"""Monocular 3D hands in camera (head) coordinates from one head-mounted fisheye video.

Per detected hand, MediaPipe gives 21 keypoints in the image AND the same 21 points in metres in a hand-centred
frame ("world landmarks"; FORGE's 2D pipeline discards them). Each 2D keypoint is turned into a viewing ray with an
equidistant fisheye model (r = f * theta; f from the horizontal field of view), and the metric hand is posed on
those rays with PnP. Result: the 21 keypoints in metres relative to the camera, a wrist orientation frame, grip
aperture, distance and speed.

What this is NOT: calibrated. The clips have no intrinsics, so `hands3d.hfov_deg` is an estimate
(docs/DATA_REPORT.md: 150 deg+), and the hand size is MediaPipe's prior. Directions are reliable, distances are
approximate (they scale with both). The camera is the origin and moves with the head; without IMU or SLAM there is
no head trajectory in the room. Coordinates: X right, Y forward (out of the lens), Z up, metres.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from egovision.config import Config, Hands3DConfig, IngestConfig

log = logging.getLogger(__name__)
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17
MAX_THETA = math.radians(85)  # rays beyond this are clamped (tan explodes at 90 deg)
COORDS = "metres; origin = camera on the head; X right, Y forward (out of the lens), Z up"


def focal_px(width: int, hfov_deg: float) -> float:
    """Equidistant fisheye: a point at angle theta from the optical axis lands r = f * theta from the centre."""
    return (width / 2) / math.radians(hfov_deg / 2)


def rays(kp2d: np.ndarray, width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pixel keypoints (N, 2) -> normalised pinhole coordinates (x/z, y/z) of their viewing rays (OpenCV axes)."""
    f = focal_px(width, hfov_deg)
    d = np.asarray(kp2d, np.float64) - np.array([width / 2, height / 2])
    r = np.hypot(d[:, 0], d[:, 1])
    theta = np.clip(r / f, 0.0, MAX_THETA)
    scale = np.where(r > 1e-9, np.tan(theta) / np.maximum(r, 1e-9), 1.0 / f)
    return d * scale[:, None]


def project(cam: np.ndarray, width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Inverse of `rays`: camera-frame points (N, 3, OpenCV axes) -> fisheye pixels (N, 2)."""
    cam = np.asarray(cam, np.float64)
    theta = np.arccos(np.clip(cam[:, 2] / np.linalg.norm(cam, axis=1), -1, 1))
    xy = cam[:, :2]
    n = np.maximum(np.linalg.norm(xy, axis=1), 1e-12)
    return np.array([width / 2, height / 2]) + (focal_px(width, hfov_deg) * theta / n)[:, None] * xy


def fit_hand(world: np.ndarray, ray_xy: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Pose MediaPipe's metric hand (21, 3) on the rays (21, 2). Returns camera-frame points (OpenCV axes: x right,
    y down, z forward) and the mean angle between fitted points and rays in degrees, or None if the fit fails."""
    obj = np.ascontiguousarray(world, np.float64)
    img = np.ascontiguousarray(ray_xy, np.float64)
    k = np.eye(3)
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, None, rvec, tvec)
    cam = obj @ cv2.Rodrigues(rvec)[0].T + tvec.reshape(1, 3)
    if (cam[:, 2] <= 0).any():
        return None
    want = np.hstack([img, np.ones((len(img), 1))])
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    got = cam / np.linalg.norm(cam, axis=1, keepdims=True)
    err = float(np.degrees(np.arccos(np.clip((want * got).sum(axis=1), -1, 1))).mean())
    return cam, err


def to_head(cam: np.ndarray) -> np.ndarray:
    """OpenCV camera axes (x right, y down, z forward) -> X right, Y forward, Z up."""
    return np.stack([cam[:, 0], cam[:, 2], -cam[:, 1]], axis=1)


def wrist_axes(kp: np.ndarray) -> np.ndarray:
    """Orthonormal wrist frame (rows): x across the knuckles (pinky -> index), y along the hand (wrist -> middle
    knuckle), z = x cross y (palm normal)."""
    x = kp[INDEX_MCP] - kp[PINKY_MCP]
    y = kp[MIDDLE_MCP] - kp[WRIST]
    x = x / max(np.linalg.norm(x), 1e-9)
    z = np.cross(x, y)
    z = z / max(np.linalg.norm(z), 1e-9)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def _r(a: Any, nd: int = 4) -> Any:
    return np.round(np.asarray(a, np.float64), nd).tolist()


def hand_record(side: str, score: float, kp_head: np.ndarray, err: float) -> dict[str, Any]:
    return {"side": side, "score": round(score, 3), "kp": _r(kp_head), "wrist": _r(kp_head[WRIST]),
            "axes": _r(wrist_axes(kp_head)), "dist_m": round(float(np.linalg.norm(kp_head[WRIST])), 4),
            "aperture_m": round(float(np.linalg.norm(kp_head[THUMB_TIP] - kp_head[INDEX_TIP])), 4),
            "fit_err_deg": round(err, 2), "speed_mps": None}


def smooth_and_speed(frames: list[dict[str, Any]], cfg: Hands3DConfig) -> None:
    """Per side: median-filter the hand position over neighbouring detections (the whole hand is shifted, its shape
    kept) and compute wrist speed from the neighbours on both sides. Raw positions stay in `raw_wrist`."""
    for side in ("left", "right"):
        seq = [(fi, h) for fi, fr in enumerate(frames) for h in fr["hands"] if h["side"] == side]
        ts = np.array([frames[fi]["t"] for fi, _ in seq])
        raw = np.array([h["wrist"] for _, h in seq]) if seq else np.zeros((0, 3))
        smooth = raw.copy()
        for i in range(len(seq)):
            # symmetric window (it keeps steady motion unbiased), shrunk at run edges and across time gaps
            h = min(cfg.smooth_window // 2, i, len(seq) - 1 - i)
            while h > 0 and (ts[i] - ts[i - h] > cfg.max_gap_s * h or ts[i + h] - ts[i] > cfg.max_gap_s * h):
                h -= 1
            smooth[i] = np.median(raw[i - h:i + h + 1], axis=0)
        for i, (_, h) in enumerate(seq):
            shift = smooth[i] - raw[i]
            h["raw_wrist"] = _r(raw[i])
            h["kp"] = _r(np.asarray(h["kp"]) + shift)
            h["wrist"] = _r(smooth[i])
            h["dist_m"] = round(float(np.linalg.norm(smooth[i])), 4)
        for i, (_, h) in enumerate(seq):
            p = i - 1 if i > 0 and ts[i] - ts[i - 1] <= cfg.max_gap_s else i
            n = i + 1 if i + 1 < len(seq) and ts[i + 1] - ts[i] <= cfg.max_gap_s else i
            if n != p and ts[n] > ts[p]:
                h["speed_mps"] = round(float(np.linalg.norm(smooth[n] - smooth[p]) / (ts[n] - ts[p])), 3)


def compute(video: str | Path, cfg: Config, progress: Callable[[float], None] | None = None,
            max_seconds: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the hand model on every `hands3d.stride`-th frame and fit each hand in 3D."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    from egovision.hands.mediapipe_backend import points_at_camera
    from egovision.ingest import FrameSource, probe
    from egovision.schema import Hand

    c, hc = cfg.hands3d, cfg.hands
    if not hc.model_path.is_file():
        raise FileNotFoundError(f"{hc.model_path} missing; run `python scripts/download_weights.py` first")
    meta = probe(video)
    src = FrameSource(meta, IngestConfig(stride=c.stride, width=c.width, cache_frames=False, max_seconds=max_seconds))
    W, H = meta.width, meta.height
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(hc.model_path)), running_mode=vision.RunningMode.VIDEO,
        num_hands=hc.num_hands, min_hand_detection_confidence=hc.min_detection_confidence,
        min_hand_presence_confidence=hc.min_presence_confidence, min_tracking_confidence=hc.min_tracking_confidence)
    counts = {"detections": 0, "fitted": 0, "rejected_fit": 0, "rejected_distance": 0, "relabelled_side": 0,
              "dropped_duplicate_side": 0,
              "dropped_other_person": 0}
    frames: list[dict[str, Any]] = []
    t0, last_ts, n_total = time.perf_counter(), -1, max(len(src), 1)
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for fr in src:
            ts = max(int(round(fr.t * 1000)), last_ts + 1)
            last_ts = ts
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(fr.image, cv2.COLOR_BGR2RGB)), ts)
            best: dict[str, dict[str, Any]] = {}
            for pts, wpts, handed in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness, strict=True):
                counts["detections"] += 1
                side = handed[0].category_name.lower()
                if hc.swap_handedness:
                    side = {"left": "right", "right": "left"}[side]
                kp2d = np.array([[p.x * W, p.y * H] for p in pts])
                if hc.filter_other_person:
                    clamped = np.clip(kp2d, 0, [W, H])
                    probe_hand = Hand(side=side, score=0.5, kp2d=clamped.tolist(),  # type: ignore[arg-type]
                                      bbox=[*clamped.min(axis=0), *clamped.max(axis=0)])
                    if points_at_camera(probe_hand, H, hc):
                        counts["dropped_other_person"] += 1
                        continue
                fit = fit_hand(np.array([[p.x, p.y, p.z] for p in wpts]), rays(kp2d, W, H, c.hfov_deg))
                if fit is None or fit[1] > c.max_fit_error_deg:
                    counts["rejected_fit"] += 1
                    continue
                kp = to_head(fit[0])
                dist = float(np.linalg.norm(kp[WRIST]))
                if not c.min_dist_m <= dist <= c.max_dist_m:
                    counts["rejected_distance"] += 1
                    continue
                rec = hand_record(side, float(handed[0].score), kp, fit[1])
                rec["kp2d"] = np.round(kp2d, 1).tolist()  # pixels, for the Studio's 2D overlay on the video
                rec["bbox"] = [round(float(x), 1) for x in (*kp2d.min(axis=0), *kp2d.max(axis=0))]
                other = "right" if side == "left" else "left"
                if side in best and other not in best:
                    # Both hands got the same label: the wearer's left hand is the one further left (FORGE D4).
                    a, b = sorted([best[side], rec], key=lambda h: h["wrist"][0])
                    a["side"], b["side"] = "left", "right"
                    best = {"left": a, "right": b}
                    counts["relabelled_side"] += 1
                    continue
                if side in best:
                    counts["dropped_duplicate_side"] += 1
                    if rec["score"] <= best[side]["score"]:
                        continue
                best[side] = rec
            counts["fitted"] += len(best)
            frames.append({"frame": fr.index, "t": round(fr.t, 3), "hands": sorted(best.values(), key=lambda h: h["side"])})
            if progress is not None and len(frames) % 50 == 0:
                progress(len(frames) / n_total)
    smooth_and_speed(frames, c)
    with_hand = [f for f in frames if f["hands"]]
    errs = [h["fit_err_deg"] for f in frames for h in f["hands"]]
    summary = {
        "video": meta.name, "duration_s": meta.duration_s, "frames": len(frames),
        "rate_hz": round(meta.fps / c.stride, 2), "frames_with_hand_pct": round(100 * len(with_hand) / max(len(frames), 1), 1),
        "counts": counts, "median_fit_err_deg": round(float(np.median(errs)), 2) if errs else None,
        "median_dist_m": {s: round(float(np.median(d)), 3) for s in ("left", "right")
                          if (d := [h["dist_m"] for f in frames for h in f["hands"] if h["side"] == s])},
        "camera": {"lens": "equidistant fisheye (assumed)", "hfov_deg": c.hfov_deg,
                   "focal_px": round(focal_px(W, c.hfov_deg), 1), "width": W, "height": H},
        "coords": COORDS,
        "note": "Approximate monocular 3D: no camera intrinsics, hand size from MediaPipe's prior. Directions are "
                "reliable; distances scale with the assumed field of view. The camera moves with the head.",
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    log.info("hands3d: %s frames, %s%% with a hand, median fit error %s deg", len(frames),
             summary["frames_with_hand_pct"], summary["median_fit_err_deg"])
    return frames, summary


def write(frames: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> Path:
    """hands3d.jsonl (one line per processed frame) + hands3d_summary.json, written atomically."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hands3d.jsonl"
    tmp = path.with_name(".hands3d.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr, separators=(",", ":")) + "\n")
    tmp.replace(path)
    (out_dir / "hands3d_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return path
