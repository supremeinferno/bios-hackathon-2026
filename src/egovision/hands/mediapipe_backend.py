"""MediaPipe Hand Landmarker (VIDEO mode) backend. Baseline per PDF step 3B."""

from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

from egovision.config import HandsConfig
from egovision.ingest import Frame
from egovision.schema import Hand, VideoMeta

_SWAP = {"left": "right", "right": "left"}


class MediaPipeHands:
    name = "mediapipe"

    def __init__(self, cfg: HandsConfig, meta: VideoMeta):
        if not cfg.model_path.is_file():
            raise FileNotFoundError(
                f"{cfg.model_path} missing; run `python scripts/download_weights.py` first"
            )
        self.cfg = cfg
        self.meta = meta
        self._lm: vision.HandLandmarker | None = None
        self._last_ts = -1

    def __enter__(self) -> MediaPipeHands:
        opts = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(self.cfg.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.cfg.num_hands,
            min_hand_detection_confidence=self.cfg.min_detection_confidence,
            min_hand_presence_confidence=self.cfg.min_presence_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._lm is not None:
            self._lm.close()
            self._lm = None

    def process(self, frame: Frame) -> list[Hand]:
        assert self._lm is not None, "use as a context manager"
        ts = int(round(frame.t * 1000))
        ts = max(ts, self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        W, H = self.meta.width, self.meta.height  # landmarks are normalised -> original pixels
        hands: list[Hand] = []
        for pts, handed in zip(res.hand_landmarks, res.handedness, strict=True):
            kp = [[min(max(p.x, 0.0), 1.0) * W, min(max(p.y, 0.0), 1.0) * H] for p in pts]
            xs = [k[0] for k in kp]
            ys = [k[1] for k in kp]
            side = handed[0].category_name.lower()
            if self.cfg.swap_handedness:
                side = _SWAP[side]
            hands.append(Hand(side=side, score=float(handed[0].score),
                              bbox=[min(xs), min(ys), max(xs), max(ys)], kp2d=kp))
        return hands
