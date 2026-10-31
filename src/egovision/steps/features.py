"""Keyframe features shared by step backends: per-side presence, contact, speed and a visual embedding."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from egovision.schema import Contact, FrameHands, Hand, ObjectDet

log = logging.getLogger(__name__)
SIDES = ("left", "right")


@dataclass
class SideFeat:
    present: bool
    hand_id: int | None = None
    contact: Contact | None = None
    speed: float = 0.0  # wrist speed in hand-sizes per second
    crop_box: list[float] | None = None  # original pixels
    emb: np.ndarray | None = None


@dataclass
class KeyframeFeatures:
    frame: int
    t: float
    objects: list[ObjectDet] = field(default_factory=list)
    sides: dict[str, SideFeat] = field(default_factory=dict)


def main_hand(fh: FrameHands, side: str) -> Hand | None:
    cands = [h for h in fh.hands if h.side == side]
    return max(cands, key=lambda h: h.score) if cands else None


def hand_speed(hands: list[FrameHands], frame: int, side: str, lag: int, fps: float) -> float:
    a = main_hand(hands[frame], side)
    b = main_hand(hands[max(0, frame - lag)], side) if frame - lag >= 0 else None
    if a is None or b is None or lag == 0:
        return 0.0
    size = max(a.bbox[2] - a.bbox[0], a.bbox[3] - a.bbox[1], 1.0)
    d = math.hypot(a.kp2d[0][0] - b.kp2d[0][0], a.kp2d[0][1] - b.kp2d[0][1])
    return d / size / (lag / fps)


def crop_box(hand: Hand, contact: Contact | None, pad: float, min_px: float, w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = hand.bbox
    if contact is not None and contact.state == "portable" and contact.object_bbox is not None:
        ob = contact.object_bbox
        x1, y1, x2, y2 = min(x1, ob[0]), min(y1, ob[1]), max(x2, ob[2]), max(y2, ob[3])
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1, min_px) * (1 + pad) / 2
    return [max(0.0, cx - half), max(0.0, cy - half), min(float(w), cx + half), min(float(h), cy + half)]


def cut(img: np.ndarray, box: list[float], to_proc: float, size: int = 224) -> np.ndarray:
    x1, y1, x2, y2 = (int(round(v * to_proc)) for v in box)
    x2, y2 = max(x2, x1 + 2), max(y2, y1 + 2)
    crop = img[max(0, y1):y2, max(0, x1):x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _tensor(out):  # transformers 5 returns a ModelOutput, 4.x a tensor
    return out.pooler_output if hasattr(out, "pooler_output") else out


class SiglipEmbedder:
    """Image/text embeddings (L2-normalised) from a SigLIP checkpoint."""

    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=self.dtype).to(device).eval()
        log.info("steps: embedder %s on %s", model_name, device)

    def images(self, crops_bgr: list[np.ndarray], batch: int = 32) -> np.ndarray:
        import torch

        outs = []
        for i in range(0, len(crops_bgr), batch):
            rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr[i:i + batch]]
            inp = self.processor(images=rgb, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                e = _tensor(self.model.get_image_features(pixel_values=inp["pixel_values"].to(self.dtype)))
            outs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 1), np.float32)

    def texts(self, texts: list[str]) -> np.ndarray:
        import torch

        inp = self.processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            e = _tensor(self.model.get_text_features(**inp))
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
