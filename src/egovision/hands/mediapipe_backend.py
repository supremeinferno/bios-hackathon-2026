"""MediaPipe Hand Landmarker (VIDEO mode) backend. Baseline per PDF step 3B, plus two optional FORGE
Stage 3 additions, both OFF by default after measurement (docs/DECISIONS.md D12, D16):
  * recovery: when fewer hands than the worker profile expects are found, retry the frame once with a
    separate IMAGE-mode landmarker; extra hands are `recovered`. Two sources (`hands.recover_detector`):
      - "clahe": run it on a contrast-enhanced (CLAHE) copy of the whole frame
      - "rtdetr": NVIDIA's RT-DETR hand detector proposes boxes; MediaPipe runs on a padded square crop
        of each box and its keypoints are mapped back. A box without a MediaPipe hand is discarded.
    The full-frame MediaPipe hands are never modified or removed.
  * other-person filter: drop hands whose fingers point down toward the camera from the upper part of
    the frame (a coworker facing the wearer); the wearer's own arms enter from below
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

from egovision.config import HandsConfig
from egovision.hands.rtdetr import accept, crop_to_frame, square_crop
from egovision.ingest import Frame
from egovision.schema import Hand, VideoMeta

if TYPE_CHECKING:
    from egovision.hands.rtdetr import RTDetrHandDetector

_SWAP = {"left": "right", "right": "left"}


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def points_at_camera(h: Hand, height: int, cfg: HandsConfig) -> bool:
    """Coworker heuristic: wrist in the upper part of the frame and fingers pointing down (toward us)."""
    wx, wy = h.kp2d[0]
    mx, my = h.kp2d[9]  # middle-finger knuckle
    size = max(h.bbox[2] - h.bbox[0], h.bbox[3] - h.bbox[1], 1.0)
    pointing_down = (my - wy) > cfg.other_person_min_down * size and (my - wy) > abs(mx - wx)
    return pointing_down and wy < cfg.other_person_max_wrist_y * height


def clahe(img: np.ndarray, clip: float) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class MediaPipeHands:
    name = "mediapipe"

    def __init__(self, cfg: HandsConfig, meta: VideoMeta, expected: int = 2, gloved: bool = False):
        if not cfg.model_path.is_file():
            raise FileNotFoundError(
                f"{cfg.model_path} missing; run `python scripts/download_weights.py` first"
            )
        self.cfg = cfg
        self.meta = meta
        self.expected = expected
        # FORGE: lower thresholds when the profile says gloves are worn
        self.det_conf = max(0.1, cfg.min_detection_confidence - (cfg.glove_threshold_drop if gloved else 0.0))
        self._lm: vision.HandLandmarker | None = None
        self._retry_lm: vision.HandLandmarker | None = None
        self._detector: RTDetrHandDetector | None = None  # only for recover_detector="rtdetr"
        self._last_ts = -1
        self.counters = {"recovered": 0, "retries": 0, "dropped_other_person": 0}
        if cfg.recover and cfg.recover_detector == "rtdetr":
            self.counters.update(rtdetr_boxes=0, rtdetr_boxes_no_hand=0, rtdetr_side_disagree=0)

    def _opts(self, mode: vision.RunningMode) -> vision.HandLandmarkerOptions:
        return vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(self.cfg.model_path)),
            running_mode=mode,
            num_hands=self.cfg.num_hands,
            min_hand_detection_confidence=self.det_conf,
            min_hand_presence_confidence=self.cfg.min_presence_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )

    def __enter__(self) -> MediaPipeHands:
        self._lm = vision.HandLandmarker.create_from_options(self._opts(vision.RunningMode.VIDEO))
        if self.cfg.recover:
            self._retry_lm = vision.HandLandmarker.create_from_options(self._opts(vision.RunningMode.IMAGE))
            if self.cfg.recover_detector == "rtdetr":
                from egovision.hands.rtdetr import RTDetrHandDetector  # lazy: needs onnxruntime

                self._detector = RTDetrHandDetector(self.cfg.rtdetr_weights, self.cfg.rtdetr_score_threshold)
        return self

    def __exit__(self, *exc: object) -> None:
        for lm in (self._lm, self._retry_lm):
            if lm is not None:
                lm.close()
        self._lm = self._retry_lm = None

    def _to_hands(self, res: vision.HandLandmarkerResult, source: str) -> list[Hand]:
        W, H = self.meta.width, self.meta.height  # landmarks are normalised -> original pixels
        hands: list[Hand] = []
        for pts, handed in zip(res.hand_landmarks, res.handedness, strict=True):
            kp = [[min(max(p.x, 0.0), 1.0) * W, min(max(p.y, 0.0), 1.0) * H] for p in pts]
            xs = [k[0] for k in kp]
            ys = [k[1] for k in kp]
            side = handed[0].category_name.lower()
            if self.cfg.swap_handedness:
                side = _SWAP[side]
            hands.append(Hand(side=side, score=float(handed[0].score), source=source,  # type: ignore[arg-type]
                              bbox=[min(xs), min(ys), max(xs), max(ys)], kp2d=kp))
        return hands

    def process(self, frame: Frame) -> list[Hand]:
        assert self._lm is not None, "use as a context manager"
        ts = int(round(frame.t * 1000))
        ts = max(ts, self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        hands = self._to_hands(self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts),
                               "detected")
        if self.cfg.filter_other_person:
            kept = [h for h in hands if not points_at_camera(h, self.meta.height, self.cfg)]
            self.counters["dropped_other_person"] += len(hands) - len(kept)
            hands = kept
        if self._retry_lm is not None and len(hands) < self.expected and self.cfg.recover_detector == "rtdetr":
            self.counters["retries"] += 1
            hands += self._recover_rtdetr(frame, hands)
        elif self._retry_lm is not None and len(hands) < self.expected:
            self.counters["retries"] += 1
            enh = cv2.cvtColor(clahe(frame.image, self.cfg.recover_clahe_clip), cv2.COLOR_BGR2RGB)
            extra = self._to_hands(self._retry_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=enh)),
                                   "recovered")
            have = {h.side for h in hands}
            for h in extra:
                if h.side in have or any(_iou(h.bbox, g.bbox) > 0.3 for g in hands):
                    continue
                if self.cfg.filter_other_person and points_at_camera(h, self.meta.height, self.cfg):
                    continue
                if h.score < self.cfg.recover_min_score:
                    continue
                hands.append(h)
                have.add(h.side)
                self.counters["recovered"] += 1
        return hands

    def _recover_rtdetr(self, frame: Frame, hands: list[Hand]) -> list[Hand]:
        """Detector box -> padded square crop -> MediaPipe IMAGE mode -> keypoints in full-frame pixels."""
        assert self._detector is not None and self._retry_lm is not None
        added: list[Hand] = []
        img = frame.image
        h_px, w_px = img.shape[:2]
        for det_side, _, box in self._detector.detect(img):  # boxes in processing-resolution pixels
            self.counters["rtdetr_boxes"] += 1
            box_orig = [v / frame.scale for v in box]
            if any(_iou(box_orig, g.bbox) > self.cfg.rtdetr_max_iou for g in hands + added):
                continue  # MediaPipe (or an earlier box) already has this hand
            crop = square_crop(box, self.cfg.rtdetr_crop_pad, w_px, h_px)
            x1, y1, x2, y2 = crop
            rgb = cv2.cvtColor(np.ascontiguousarray(img[y1:y2, x1:x2]), cv2.COLOR_BGR2RGB)
            res = self._retry_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            found = [self._crop_hand(pts, handed, crop, frame.scale)
                     for pts, handed in zip(res.hand_landmarks, res.handedness, strict=True)]
            if not found:
                self.counters["rtdetr_boxes_no_hand"] += 1
                continue  # no keypoints = no hand
            best = max(found, key=lambda x: x.score)
            if best.side != det_side:
                self.counters["rtdetr_side_disagree"] += 1
            if accept(best, hands + added, self.cfg.rtdetr_max_iou, self.cfg.recover_min_score):
                added.append(best)
                self.counters["recovered"] += 1
        return added

    def _crop_hand(self, pts: list, handed: list, crop: tuple[int, int, int, int], scale: float) -> Hand:
        kp = crop_to_frame([(p.x, p.y) for p in pts], crop, scale)
        xs = [k[0] for k in kp]
        ys = [k[1] for k in kp]
        side = handed[0].category_name.lower()
        if self.cfg.swap_handedness:  # same convention as the full-frame path (D4: off)
            side = _SWAP[side]
        return Hand(side=side, score=float(handed[0].score), source="recovered",
                    bbox=[min(xs), min(ys), max(xs), max(ys)], kp2d=kp)


__all__ = ["MediaPipeHands", "clahe", "points_at_camera"]
