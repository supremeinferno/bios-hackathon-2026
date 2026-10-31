"""Simple multi-hand tracking: stable IDs, re-identification after short exits, side voting, smoothing.

Egocentric prior: the wearer has at most one left and one right hand, and a hand that leaves the
view usually comes back on the same side. So a new detection first tries to resume a recently
lost track of the same side before a new ID is created.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from egovision.config import HandsConfig
from egovision.schema import FrameHands, Hand


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    id: int
    side: str
    bbox: list[float]
    last_frame: int


def _cost(tr: _Track, h: Hand, diag: float, side_penalty: float) -> float:
    cx1, cy1 = (tr.bbox[0] + tr.bbox[2]) / 2, (tr.bbox[1] + tr.bbox[3]) / 2
    cx2, cy2 = (h.bbox[0] + h.bbox[2]) / 2, (h.bbox[1] + h.bbox[3]) / 2
    dist = math.hypot(cx1 - cx2, cy1 - cy2) / diag
    return (1 - iou(tr.bbox, h.bbox)) + min(dist, 1.0) + (side_penalty if tr.side != h.side else 0.0)


def track_hands(frames: list[FrameHands], fps: float, width: int, height: int, cfg: HandsConfig) -> list[FrameHands]:
    """Assign track IDs to per-frame detections (frames must be in time order)."""
    diag = math.hypot(width, height) * 0.25  # distance normaliser: a quarter of the image diagonal
    max_age = int(round(cfg.track_max_age_s * fps))
    reid = int(round(cfg.reid_window_s * fps))
    active: list[_Track] = []
    lost: list[_Track] = []
    next_id = 0
    out: list[FrameHands] = []
    for fh in frames:
        still = [t for t in active if fh.frame - t.last_frame <= max_age]
        lost += [t for t in active if fh.frame - t.last_frame > max_age]
        active = still
        lost = [t for t in lost if fh.frame - t.last_frame <= reid]
        dets = list(fh.hands)
        assigned: dict[int, _Track] = {}
        if active and dets:
            cost = np.array([[_cost(t, d, diag, cfg.track_side_penalty) for d in dets] for t in active])
            for r, c in zip(*linear_sum_assignment(cost), strict=True):
                if cost[r, c] <= cfg.track_max_cost:
                    assigned[c] = active[r]
        for j, d in enumerate(dets):
            if j in assigned:
                continue
            busy = {t.id for t in assigned.values()} | {t.id for t in active}
            cand = [t for t in lost if t.side == d.side and t.id not in busy]
            if cand:
                tr = max(cand, key=lambda t: t.last_frame)
                lost.remove(tr)
            else:
                tr = _Track(id=next_id, side=d.side, bbox=d.bbox, last_frame=fh.frame)
                next_id += 1
            active.append(tr)
            assigned[j] = tr
        hands = []
        for j, d in enumerate(dets):
            tr = assigned[j]
            tr.bbox, tr.last_frame = d.bbox, fh.frame
            hands.append(d.model_copy(update={"id": tr.id}))
        out.append(FrameHands(frame=fh.frame, t=fh.t, hands=hands))
    return vote_sides(out) if cfg.side_vote else out


def vote_sides(frames: list[FrameHands]) -> list[FrameHands]:
    """Give every detection in a track the track's score-weighted majority side (fixes L/R flicker)."""
    votes: dict[int, dict[str, float]] = defaultdict(lambda: {"left": 0.0, "right": 0.0})
    for fh in frames:
        for h in fh.hands:
            if h.id is not None:
                votes[h.id][h.side] += h.score
    side = {i: max(v, key=v.__getitem__) for i, v in votes.items()}
    return [FrameHands(frame=fh.frame, t=fh.t,
                       hands=[h.model_copy(update={"side": side.get(h.id, h.side)}) for h in fh.hands])
            for fh in frames]


class OneEuro:
    """One-Euro filter on numpy arrays (PDF step 4): smooths more when still, less when moving fast."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t: float | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None or self.t is None or self.dx is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-6)
        self.t = t
        self.dx = self.dx + self._alpha(self.dc, dt) * ((x - self.x) / dt - self.dx)
        cutoff = self.mc + self.beta * np.abs(self.dx)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        return self.x


def smooth_tracks(frames: list[FrameHands], cfg: HandsConfig, reset_gap_s: float = 0.5) -> list[FrameHands]:
    """Per-track One-Euro smoothing of keypoints; the box is recomputed from smoothed points."""
    filters: dict[int, OneEuro] = {}
    last_t: dict[int, float] = {}
    out = []
    for fh in frames:
        hands = []
        for h in fh.hands:
            if h.id is None:
                hands.append(h)
                continue
            if h.id not in filters or fh.t - last_t[h.id] > reset_gap_s:
                filters[h.id] = OneEuro(cfg.smooth_min_cutoff, cfg.smooth_beta)
            last_t[h.id] = fh.t
            kp = filters[h.id](np.asarray(h.kp2d, dtype=np.float64), fh.t)
            hands.append(h.model_copy(update={
                "kp2d": kp.tolist(),
                "bbox": [float(kp[:, 0].min()), float(kp[:, 1].min()), float(kp[:, 0].max()), float(kp[:, 1].max())],
            }))
        out.append(FrameHands(frame=fh.frame, t=fh.t, hands=hands))
    return out
