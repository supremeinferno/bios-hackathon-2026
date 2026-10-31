"""Open-vocabulary object detection (OWLv2 or Grounding DINO via Hugging Face transformers)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from egovision.config import ObjectsConfig, VocabItem
from egovision.schema import ObjectDet

log = logging.getLogger(__name__)


class OpenVocabDetector:
    def __init__(self, cfg: ObjectsConfig, device: str):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.cfg = cfg
        self.device = device
        self.family = "grounding-dino" if "grounding" in cfg.model.lower() else "owl"
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(cfg.model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model, dtype=self.dtype)
        self.model.to(device).eval()
        log.info("objects: %s (%s) on %s", cfg.model, self.family, device)

    def detect(self, images: list[np.ndarray], vocab: list[VocabItem], to_orig: float) -> list[list[ObjectDet]]:
        """images: BGR at processing resolution; boxes are returned in original-video pixels."""
        import torch

        out: list[list[ObjectDet]] = []
        for i in range(0, len(images), self.cfg.batch_size):
            batch = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images[i:i + self.cfg.batch_size]]
            with torch.inference_mode():
                raw = self._owl(batch, vocab) if self.family == "owl" else self._gdino(batch, vocab)
            for dets, im in zip(raw, batch, strict=True):
                out.append(self._filter(dets, vocab, im.shape[1], im.shape[0], to_orig))
        return out

    def _owl(self, batch: list[np.ndarray], vocab: list[VocabItem]) -> list[list[tuple[int, float, list[float]]]]:
        import torch

        prompts = [[f"a photo of a {v.prompt}" for v in vocab]] * len(batch)
        inputs = self.processor(text=prompts, images=batch, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        outputs = self.model(**inputs)
        # OWLv2 pads images to a square at the bottom/right: boxes are relative to that square.
        sizes = torch.tensor([[max(im.shape[:2])] * 2 for im in batch], device=self.device)
        res = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=self.cfg.score_threshold, target_sizes=sizes)
        return [[(int(lab), float(s), b.tolist()) for s, lab, b in zip(r["scores"], r["labels"], r["boxes"], strict=True)]
                for r in res]

    def _gdino(self, batch: list[np.ndarray], vocab: list[VocabItem]) -> list[list[tuple[int, float, list[float]]]]:
        text = ". ".join(v.prompt for v in vocab) + "."
        inputs = self.processor(images=batch, text=[text] * len(batch), return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=self.cfg.score_threshold, text_threshold=self.cfg.score_threshold,
            target_sizes=[im.shape[:2] for im in batch])
        prompts = [v.prompt.lower() for v in vocab]
        out = []
        for r in res:
            dets = []
            labels = r.get("text_labels", r.get("labels"))
            for s, lab, b in zip(r["scores"], labels, r["boxes"], strict=True):
                lab = str(lab).lower().strip()
                idx = next((k for k, p in enumerate(prompts) if lab and (lab in p or p in lab)), None)
                if idx is not None:
                    dets.append((idx, float(s), b.tolist()))
            out.append(dets)
        return out

    def _filter(self, dets: list[tuple[int, float, list[float]]], vocab: list[VocabItem], w: int, h: int,
                to_orig: float) -> list[ObjectDet]:
        kept: list[ObjectDet] = []
        for idx, score, box in sorted(dets, key=lambda d: -d[1]):
            x1, y1, x2, y2 = (max(0.0, min(v, lim)) for v, lim in zip(box, (w, h, w, h), strict=True))
            if x2 <= x1 or y2 <= y1:
                continue
            item = vocab[idx]
            if item.kind == "portable" and (x2 - x1) * (y2 - y1) > self.cfg.max_area_frac * w * h:
                continue  # a "portable" box covering a third of the frame is almost always wrong
            box_o = [x1 * to_orig, y1 * to_orig, x2 * to_orig, y2 * to_orig]
            if any(k.label == item.name and _iou(k.bbox, box_o) > self.cfg.nms_iou for k in kept):
                continue  # per-class NMS
            kept.append(ObjectDet(label=item.name, kind=item.kind, score=round(min(score, 1.0), 3), bbox=box_o))
        return kept


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
