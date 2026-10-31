from __future__ import annotations

from egovision.config import HandsConfig
from egovision.hands.mediapipe_backend import points_at_camera
from egovision.schema import Hand

H = 1080


def hand(wrist, knuckle):
    kp = [list(wrist)] * 21
    kp[9] = list(knuckle)
    xs, ys = [p[0] for p in kp], [p[1] for p in kp]
    return Hand(side="left", score=0.9, bbox=[min(xs), min(ys), max(xs) + 1, max(ys) + 1], kp2d=kp)


def test_coworker_hand_pointing_down_from_top_is_flagged():
    cfg = HandsConfig()
    assert points_at_camera(hand((900, 200), (905, 300)), H, cfg)  # top of frame, fingers point down
    assert not points_at_camera(hand((900, 900), (905, 800)), H, cfg)  # wearer: from below, fingers up
    assert not points_at_camera(hand((900, 800), (905, 900)), H, cfg)  # pointing down but low in frame
    assert not points_at_camera(hand((900, 200), (1000, 230)), H, cfg)  # mostly sideways
