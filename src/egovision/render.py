"""Overlay renderer: hands (boxes, skeleton, ID), held objects, current step, and a step timeline."""

from __future__ import annotations

import bisect
import colorsys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from egovision.config import RenderConfig
from egovision.hands import HAND_CONNECTIONS
from egovision.ingest import iter_all_frames
from egovision.media import write_video
from egovision.schema import Action, FrameHands, FrameObjects, VideoMeta

SIDE_COLOR = {"left": (80, 220, 80), "right": (0, 150, 255)}  # BGR
OBJ_COLOR = (255, 200, 60)


def verb_colors(verbs: list[str]) -> dict[str, tuple[int, int, int]]:
    out = {}
    for i, v in enumerate(verbs):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.65, 0.95)
        out[v] = (int(b * 255), int(g * 255), int(r * 255))
    out["other"] = (150, 150, 150)
    return out


def _text(img: np.ndarray, s: str, org: tuple[int, int], col: tuple[int, int, int], scale: float = 0.55) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def timeline_image(actions: list[Action], duration: float, w: int, h: int, colors: dict) -> np.ndarray:
    """Two lanes: left hand on top, right hand below; 'both'/'none' span both lanes."""
    img = np.full((h, w, 3), 30, np.uint8)
    lane = h // 2
    for a in actions:
        x1 = int(a.start_s / duration * w)
        x2 = max(int(a.end_s / duration * w), x1 + 1)
        y1, y2 = {"left": (0, lane), "right": (lane, h)}.get(a.hand, (0, h))
        cv2.rectangle(img, (x1, y1 + 1), (x2, y2 - 1), colors.get(a.verb, (150, 150, 150)), -1)
    cv2.line(img, (0, lane), (w, lane), (60, 60, 60), 1)
    _text(img, "L", (3, lane - 8), (230, 230, 230), 0.45)
    _text(img, "R", (3, h - 8), (230, 230, 230), 0.45)
    return img


def draw_frame(img: np.ndarray, fh: FrameHands, fo: FrameObjects | None, now: list[Action], t: float,
               s: float, cfg: RenderConfig) -> None:
    held = {c.hand_id: c for c in fo.contacts} if fo else {}
    if fo and cfg.draw_objects:
        for o in fo.objects:
            x1, y1, x2, y2 = (int(v * s) for v in o.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), OBJ_COLOR, 1)
            _text(img, f"{o.label} {o.score:.2f}", (x1 + 2, y2 - 4), OBJ_COLOR, 0.4)
    for hand in fh.hands:
        col = SIDE_COLOR[hand.side]
        thick = 1 if hand.interp else 2
        kp = [(int(x * s), int(y * s)) for x, y in hand.kp2d]
        x1, y1, x2, y2 = (int(v * s) for v in hand.bbox)
        if cfg.draw_bbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
        if cfg.draw_keypoints:
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, kp[a], kp[b], col, thick, cv2.LINE_AA)
            for p in kp:
                cv2.circle(img, p, 2, (255, 255, 255), -1)
        tag = f"{hand.side[0].upper()}{'' if hand.id is None else hand.id} {hand.score:.2f}"
        c = held.get(hand.id) if hand.id is not None else None
        if c is not None and c.state != "none":
            tag += f" | {'holds' if c.state == 'portable' else 'touches'} {c.object}"
            if c.object_bbox is not None:
                ox1, oy1, ox2, oy2 = (int(v * s) for v in c.object_bbox)
                cv2.rectangle(img, (ox1, oy1), (ox2, oy2), col, 2)
        _text(img, tag, (x1, max(14, y1 - 6)), col)
    label = "  |  ".join(f"{a.hand}: {a.verb} {a.object}" for a in now) or "-"
    _text(img, f"{t:6.2f}s  {label}", (8, 24), (255, 255, 255), 0.6)


def render_overlay(meta: VideoMeta, hands: list[FrameHands], objects: list[FrameObjects], actions: list[Action],
                   verbs: list[str], cfg: RenderConfig, out: Path, start_frame: int = 0,
                   end_frame: int | None = None) -> str:
    colors = verb_colors(verbs)
    s = cfg.width / meta.width
    duration = len(hands) / meta.fps if hands else meta.duration_s
    tl = timeline_image(actions, duration, cfg.width, cfg.timeline_height, colors) if cfg.timeline_height else None
    key_frames = [fo.frame for fo in objects]
    last = min(len(hands), end_frame if end_frame is not None else len(hands)) - 1

    def frames() -> Iterator[np.ndarray]:
        for i, img in iter_all_frames(meta.path, cfg.width, last):
            if i < start_frame:
                continue
            t = i / meta.fps
            j = bisect.bisect_right(key_frames, i) - 1
            fo = objects[j] if j >= 0 else None
            draw_frame(img, hands[i], fo, [a for a in actions if a.start_s <= t < a.end_s], t, s, cfg)
            if tl is not None:
                strip = tl.copy()
                x = int(t / duration * cfg.width)
                cv2.line(strip, (x, 0), (x, strip.shape[0]), (255, 255, 255), 2)
                img = np.vstack([img, strip])
            yield img

    return write_video(frames(), out, meta.fps)
