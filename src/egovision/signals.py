"""Stage 4 camera-origin coordinates and hand signals (FORGE).

The camera is the origin (0, 0) at the image centre: x from -1 (left) to +1 (right), y from -1
(down) to +1 (up). The camera is on the worker's head, so this is a camera-relative frame, not a
fixed workstation frame: head motion moves everything. Outputs are labelled `frame: camera`.
"""

from __future__ import annotations

import math
from typing import Any

from egovision.config import SignalsConfig
from egovision.schema import FrameHands, Hand

SIDES = ("left", "right")
COLS = ("left", "centre", "right")
ROWS = ("upper", "middle", "lower")


def to_camera_xy(u: float, v: float, width: int, height: int) -> tuple[float, float]:
    return (u - width / 2) / (width / 2), (height / 2 - v) / (height / 2)


def zone(x: float, y: float) -> str:
    col = COLS[min(2, max(0, int((x + 1) / 2 * 3)))]
    row = ROWS[min(2, max(0, int((1 - y) / 2 * 3)))]
    return "centre" if (row, col) == ("middle", "centre") else f"{row}-{col}"


def anchor(h: Hand) -> tuple[float, float]:
    """Hand position = wrist keypoint (index 0); robust to fingers opening and closing."""
    return h.kp2d[0][0], h.kp2d[0][1]


def _pick(fh: FrameHands, side: str) -> Hand | None:
    c = [h for h in fh.hands if h.side == side]
    return max(c, key=lambda h: h.score) if c else None


def build_signals(hands: list[FrameHands], fps: float, width: int, height: int, cfg: SignalsConfig,
                  qc_frames: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One row per sample at `cfg.hz`: per side visible, x, y, r, zone, speed, source; plus blur flag."""
    step = max(1, int(round(fps / cfg.hz)))
    qc_by = {q["frame"]: q for q in qc_frames or []}
    qc_keys = sorted(qc_by)
    rows: list[dict[str, Any]] = []
    prev: dict[str, tuple[float, float, float] | None] = {s: None for s in SIDES}
    for i in range(0, len(hands), step):
        fh = hands[i]
        row: dict[str, Any] = {"frame": i, "t": round(i / fps, 3), "frame_ref": "camera"}
        if qc_keys:
            import bisect

            j = bisect.bisect_right(qc_keys, i) - 1
            row["blurry"] = bool(qc_by[qc_keys[max(j, 0)]]["blurry"])
        for s in SIDES:
            h = _pick(fh, s)
            if h is None:
                row.update({f"{s}_visible": 0, f"{s}_x": "", f"{s}_y": "", f"{s}_r": "", f"{s}_zone": "",
                            f"{s}_speed": "", f"{s}_source": ""})
                prev[s] = None
                continue
            x, y = to_camera_xy(*anchor(h), width, height)
            speed = ""
            p = prev[s]
            if p is not None and fh.t > p[2]:
                speed = round(math.hypot(x - p[0], y - p[1]) / (fh.t - p[2]), 3)
            prev[s] = (x, y, fh.t)
            row.update({f"{s}_visible": 1, f"{s}_x": round(x, 4), f"{s}_y": round(y, 4),
                        f"{s}_r": round(math.hypot(x, y), 4), f"{s}_zone": zone(x, y), f"{s}_speed": speed,
                        f"{s}_source": "interpolated" if h.interp else "detected"})
        rows.append(row)
    return rows


SIGNAL_COLUMNS = ["frame", "t", "frame_ref", "blurry"] + [
    f"{s}_{k}" for s in SIDES for k in ("visible", "x", "y", "r", "zone", "speed", "source")]
