"""Overlay renderer: hand skeletons/boxes on every frame plus a step timeline strip below."""

from __future__ import annotations

import colorsys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from egovision.config import RenderConfig
from egovision.hands import HAND_CONNECTIONS
from egovision.ingest import iter_all_frames
from egovision.media import write_video
from egovision.schema import Action, FrameHands, VideoMeta

SIDE_COLOR = {"left": (80, 220, 80), "right": (0, 150, 255)}  # BGR


def verb_colors(verbs: list[str]) -> dict[str, tuple[int, int, int]]:
    out = {}
    for i, v in enumerate(verbs):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.65, 0.95)
        out[v] = (int(b * 255), int(g * 255), int(r * 255))
    out["other"] = (150, 150, 150)
    return out


def _timeline(actions: list[Action], duration: float, w: int, h: int, colors: dict) -> np.ndarray:
    """Two lanes: left hand on top, right hand below; 'both'/'none' span both lanes."""
    img = np.full((h, w, 3), 30, np.uint8)
    lane = h // 2
    for a in actions:
        x1, x2 = int(a.start_s / duration * w), max(int(a.end_s / duration * w), int(a.start_s / duration * w) + 1)
        y1, y2 = {"left": (0, lane), "right": (lane, h)}.get(a.hand, (0, h))
        cv2.rectangle(img, (x1, y1 + 1), (x2, y2 - 1), colors.get(a.verb, (150, 150, 150)), -1)
    cv2.putText(img, "L", (2, lane - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    cv2.putText(img, "R", (2, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    return img


def _draw_hands(img: np.ndarray, fh: FrameHands, s: float, cfg: RenderConfig) -> None:
    for hand in fh.hands:
        col = SIDE_COLOR[hand.side]
        thick = 1 if hand.interp else 2
        kp = [(int(x * s), int(y * s)) for x, y in hand.kp2d]
        if cfg.draw_bbox:
            x1, y1, x2, y2 = (int(v * s) for v in hand.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
            tag = f"{hand.side[0].upper()}{'' if hand.id is None else hand.id} {hand.score:.2f}"
            cv2.putText(img, tag, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if cfg.draw_keypoints:
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, kp[a], kp[b], col, thick, cv2.LINE_AA)
            for p in kp:
                cv2.circle(img, p, 2, (255, 255, 255), -1)


def render_overlay(meta: VideoMeta, hands: list[FrameHands], actions: list[Action], verbs: list[str],
                   cfg: RenderConfig, out: Path) -> str:
    colors = verb_colors(verbs)
    s = cfg.width / meta.width
    duration = len(hands) / meta.fps if hands else meta.duration_s
    tl = _timeline(actions, duration, cfg.width, cfg.timeline_height, colors) if cfg.timeline_height else None

    def frames() -> Iterator[np.ndarray]:
        for i, img in iter_all_frames(meta.path, cfg.width):
            if i >= len(hands):
                break
            t = i / meta.fps
            _draw_hands(img, hands[i], s, cfg)
            now = [a for a in actions if a.start_s <= t < a.end_s]
            label = "  |  ".join(f"{a.hand}: {a.verb} {a.object}" for a in now) or "-"
            cv2.putText(img, f"{t:6.2f}s  {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(img, f"{t:6.2f}s  {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if tl is not None:
                strip = tl.copy()
                x = int(t / duration * cfg.width)
                cv2.line(strip, (x, 0), (x, strip.shape[0]), (255, 255, 255), 2)
                img = np.vstack([img, strip])
            yield img

    return write_video(frames(), out, meta.fps)
