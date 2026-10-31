from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from egovision.schema import Hand


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """2 s, 10 fps, 160x120 clip whose frames encode their own index (for ingest tests; no hands)."""
    p = tmp_path / "synthetic.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for i in range(20):
        img = np.full((120, 160, 3), (i * 12) % 255, np.uint8)
        cv2.putText(img, str(i), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        w.write(img)
    w.release()
    return p


def make_hand(side: str = "left", x: float = 100.0, y: float = 100.0, score: float = 0.9, hid: int | None = None) -> Hand:
    kp = [[x + i, y + i] for i in range(21)]
    return Hand(id=hid, side=side, score=score, bbox=[x, y, x + 20, y + 20], kp2d=kp)  # type: ignore[arg-type]
