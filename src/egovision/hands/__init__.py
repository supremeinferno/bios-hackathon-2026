"""Hand perception stage: per-frame detection + landmarks, then temporal post-processing."""

from __future__ import annotations

from typing import Protocol

from egovision.config import HandsConfig
from egovision.ingest import Frame
from egovision.schema import Hand, VideoMeta

# MediaPipe 21-point hand topology (wrist=0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20).
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGERTIPS = [4, 8, 12, 16, 20]


class HandBackend(Protocol):
    name: str

    def __enter__(self) -> HandBackend: ...
    def __exit__(self, *exc: object) -> None: ...
    def process(self, frame: Frame) -> list[Hand]: ...


def make_backend(cfg: HandsConfig, meta: VideoMeta) -> HandBackend:
    if cfg.backend == "mediapipe":
        from egovision.hands.mediapipe_backend import MediaPipeHands

        return MediaPipeHands(cfg, meta)
    raise ValueError(f"unknown hand backend {cfg.backend!r}")
