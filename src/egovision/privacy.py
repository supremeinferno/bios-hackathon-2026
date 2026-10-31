"""Stage 1 privacy sanitization (FORGE): a pixelated copy of the clip with people's heads,
signage/logos/text, screens and documents hidden, and the wearer's hands kept visible.

Detections come from the shared OWLv2 keyframe pass (prompts in `PrivacyConfig.prompts`). Rules:
  * 'person' boxes become head bands (top `person_head_frac`), because OWLv2 finds people far
    more reliably than faces on our fisheye footage (docs/DECISIONS.md D9)
  * 'person' boxes that are the wearer's own arms/body (touch the bottom edge and cover a tracked
    wearer hand, or fill a large share of the frame from the bottom) are ignored
  * every region is padded and applied to all frames within +/- `hold_s` of its keyframe, so the
    blur does not flicker between detections
  * the wearer's current hand boxes are cut out of the mask so hands stay visible for review
Hand tracking always runs on the ORIGINAL frames; the sanitized copy is for the API, UI and demos.
Blurring is an extra safety layer: the footage stays confidential under the data-use rules.
"""

from __future__ import annotations

import bisect
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from egovision.config import PrivacyConfig
from egovision.ingest import iter_all_frames
from egovision.media import write_video
from egovision.schema import FrameHands, Hand, ObjectDet, VideoMeta

log = logging.getLogger(__name__)
PREFIX = "privacy:"


@dataclass
class Region:
    category: str  # head | face | sign | logo | text | screen | document
    box: list[float]  # original pixels, padded
    score: float
    source: str  # detector label it came from


@dataclass
class KeyRegions:
    frame: int
    t: float
    regions: list[Region] = field(default_factory=list)


def _area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: list[float], b: list[float]) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def pad(box: list[float], frac: float, w: int, h: int) -> list[float]:
    dx, dy = (box[2] - box[0]) * frac / 2, (box[3] - box[1]) * frac / 2
    return [max(0.0, box[0] - dx), max(0.0, box[1] - dy), min(float(w), box[2] + dx), min(float(h), box[3] + dy)]


def is_wearer_body(box: list[float], hands: list[Hand], w: int, h: int) -> bool:
    """A 'person' box anchored at the bottom edge that covers a wearer hand, or is huge, is the wearer."""
    if box[3] < 0.9 * h:
        return False
    if _area(box) > 0.35 * w * h:
        return True
    return any(_inter(box, hd.bbox) > 0.3 * max(_area(hd.bbox), 1.0) for hd in hands)


def regions_from_detections(dets: list[ObjectDet], hands: list[Hand], w: int, h: int,
                            cfg: PrivacyConfig) -> list[Region]:
    out = []
    for d in dets:
        if d.kind != "privacy" or not d.label.startswith(PREFIX):
            continue
        cat = d.label[len(PREFIX):]
        box = list(d.bbox)
        if cat == "person":
            if cfg.drop_wearer_arms and is_wearer_body(box, hands, w, h):
                continue
            box = [box[0], box[1], box[2], box[1] + cfg.person_head_frac * (box[3] - box[1])]
            cat = "head"
        if cat in ("head", "face") and box[1] > cfg.head_max_top_frac * h:
            continue
        out.append(Region(category=cat, box=pad(box, cfg.pad_frac, w, h), score=d.score, source=d.label))
    return out


def hand_cutouts(hands: list[Hand], frac: float, w: int, h: int) -> list[list[float]]:
    return [pad(hd.bbox, frac, w, h) for hd in hands]


def pixelate(img: np.ndarray, block: int) -> np.ndarray:
    hh, ww = img.shape[:2]
    small = cv2.resize(img, (max(1, ww // block), max(1, hh // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (ww, hh), interpolation=cv2.INTER_NEAREST)


ROUND = ("head", "face")  # pixelated as ellipses: tighter around a head than a rectangle


def frame_mask(shape: tuple[int, int], boxes: list[list[float]], keep: list[list[float]], s: float,
               ellipse: list[bool] | None = None) -> np.ndarray:
    """Binary mask (1 = pixelate) at output scale `s`; `keep` boxes are cut out afterwards."""
    m = np.zeros(shape, np.uint8)
    for k, b in enumerate(boxes):
        x1, y1, x2, y2 = (int(round(v * s)) for v in b)
        if ellipse is not None and ellipse[k]:
            cv2.ellipse(m, ((x1 + x2) // 2, (y1 + y2) // 2), (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)),
                        0, 0, 360, 1, -1)
        else:
            m[max(0, y1):y2, max(0, x1):x2] = 1
    for b in keep:
        x1, y1, x2, y2 = (int(round(v * s)) for v in b)
        m[max(0, y1):y2, max(0, x1):x2] = 0
    return m


class Sanitizer:
    def __init__(self, meta: VideoMeta, cfg: PrivacyConfig):
        self.meta = meta
        self.cfg = cfg
        self.keys: list[KeyRegions] = []

    def add(self, frame: int, t: float, dets: list[ObjectDet], hands: list[Hand]) -> None:
        self.keys.append(KeyRegions(frame, t, regions_from_detections(dets, hands, self.meta.width,
                                                                      self.meta.height, self.cfg)))

    def active(self, t: float) -> list[Region]:
        """All regions from keyframes within +/- hold_s of time t."""
        times = [k.t for k in self.keys]
        lo = bisect.bisect_left(times, t - self.cfg.hold_s)
        hi = bisect.bisect_right(times, t + self.cfg.hold_s)
        return [r for k in self.keys[lo:hi] for r in k.regions]

    def write(self, hands: list[FrameHands], out_path: Path) -> dict[str, Any]:
        self.keys.sort(key=lambda k: k.t)
        n = len(hands)
        w_out = min(self.cfg.output_width, self.meta.width)
        s = w_out / self.meta.width
        stats = {"frames": 0, "frames_blurred": 0, "area_sum": 0.0, "frames_hands_protected": 0}

        def frames() -> Iterator[np.ndarray]:
            for i, img in iter_all_frames(self.meta.path, w_out, n - 1):
                stats["frames"] += 1
                regions = self.active(i / self.meta.fps)
                if not regions:
                    yield img
                    continue
                keep = hand_cutouts(hands[i].hands, self.cfg.hand_pad_frac, self.meta.width, self.meta.height) \
                    if self.cfg.protect_hands else []
                m = frame_mask(img.shape[:2], [r.box for r in regions], [], s,
                               [r.category in ROUND for r in regions])
                before = int(m.sum())
                for b in keep:
                    x1, y1, x2, y2 = (int(round(v * s)) for v in b)
                    m[max(0, y1):y2, max(0, x1):x2] = 0
                if keep and int(m.sum()) < before:
                    stats["frames_hands_protected"] += 1
                if m.any():
                    stats["frames_blurred"] += 1
                    stats["area_sum"] += float(m.mean())
                    img = np.where(m[..., None].astype(bool), pixelate(img, self.cfg.pixel_block), img)
                yield img

        enc = write_video(frames(), out_path, self.meta.fps, bitrate=2_000_000)
        cats = Counter(r.category for k in self.keys for r in k.regions)
        f = max(stats["frames"], 1)
        return {
            "sanitized_path": str(out_path).replace("\\", "/"), "encoder": enc, "width": w_out,
            "keyframes": len(self.keys), "keyframes_with_regions": sum(bool(k.regions) for k in self.keys),
            "regions_by_category": dict(sorted(cats.items())),
            "frames_blurred_pct": round(100 * stats["frames_blurred"] / f, 1),
            "mean_blurred_area_pct": round(100 * stats["area_sum"] / max(stats["frames_blurred"], 1), 1),
            "frames_with_hands_protected": stats["frames_hands_protected"],
            "note": "Pixelation is an extra safety layer; the footage is still confidential (data-use rules).",
        }

    def log_record(self, summary: dict[str, Any]) -> dict[str, Any]:
        return {**summary, "config": self.cfg.model_dump(mode="json"),
                "keyframe_regions": [{"frame": k.frame, "t": k.t,
                                      "regions": [{"category": r.category, "box": [round(v, 1) for v in r.box],
                                                   "score": round(r.score, 3)} for r in k.regions]}
                                     for k in self.keys if k.regions]}
