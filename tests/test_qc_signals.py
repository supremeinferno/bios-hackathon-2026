from __future__ import annotations

import numpy as np
import pytest

from egovision.config import QCConfig, SignalsConfig
from egovision.qc import frame_qc, summarize
from egovision.schema import FrameHands
from egovision.signals import build_signals, to_camera_xy, zone

from .conftest import make_hand


def test_camera_xy_and_zones():
    assert to_camera_xy(960, 540, 1920, 1080) == (0.0, 0.0)
    assert to_camera_xy(0, 1080, 1920, 1080) == (-1.0, -1.0)  # bottom-left
    assert to_camera_xy(1920, 0, 1920, 1080) == (1.0, 1.0)  # top-right
    assert zone(0, 0) == "centre" and zone(-0.9, -0.9) == "lower-left" and zone(0.9, 0.9) == "upper-right"
    assert zone(1.0, -1.0) == "lower-right"  # edges clamp into the grid


def test_qc_relative_blur_and_covered():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
    blurry = np.full((180, 320, 3), 128, np.uint8)
    blurry[:, :160] = 140
    dark = np.full((180, 320, 3), 5, np.uint8)
    raw = [frame_qc(im, 320) for im in (sharp, sharp, sharp, blurry, dark)]
    q = summarize([0, 2, 4, 6, 8], 30.0, raw, QCConfig())
    assert [f["blurry"] for f in q["frames"]] == [False, False, False, True, True]
    assert [f["covered"] for f in q["frames"]] == [False, False, False, False, True]
    assert q["summary"]["blurry_pct"] == pytest.approx(40.0)


def test_signals_rows_speed_and_missing_side():
    frames = [FrameHands(frame=i, t=i / 30, hands=[make_hand("left", x=960 + 3 * i, y=540, hid=0)]) for i in range(9)]
    rows = build_signals(frames, 30.0, 1920, 1080, SignalsConfig(hz=10))
    assert [r["frame"] for r in rows] == [0, 3, 6]
    assert rows[0]["left_visible"] == 1 and rows[0]["right_visible"] == 0 and rows[0]["left_zone"] == "centre"
    assert rows[0]["left_speed"] == "" and rows[1]["left_speed"] == pytest.approx((9 / 960) / 0.1, abs=1e-3)  # rounded to 3 dp
