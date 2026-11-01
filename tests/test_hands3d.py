from __future__ import annotations

import math

import cv2
import numpy as np

from egovision.config import Hands3DConfig
from egovision.hands3d import fit_hand, focal_px, project, rays, smooth_and_speed, to_head, wrist_axes

# A plausible metric hand (21 MediaPipe joints, metres, hand-centred): wrist, then 4 joints per finger.
HAND = np.array([[0, 0.04, 0]] + [[(f - 2) * 0.02 + 0.004 * j * (f - 2), -0.01 - 0.025 * j, 0.004 * j * (f % 2)]
                                  for f in range(5) for j in range(1, 5)], np.float64)


def test_fisheye_rays_round_trip():
    pts = np.array([[0.0, 0.0, 1.0], [0.3, -0.2, 0.4], [-0.5, 0.4, 0.2]])  # incl. 68 deg off-axis
    px = project(pts, 1920, 1080, 150.0)
    r = rays(px, 1920, 1080, 150.0)
    np.testing.assert_allclose(r, pts[:, :2] / pts[:, 2:3], rtol=1e-9, atol=1e-12)
    assert math.isclose(focal_px(1920, 150.0), 960 / math.radians(75))


def test_fit_recovers_known_hand_pose():
    rot = cv2.Rodrigues(np.array([0.4, -0.3, 0.2]))[0]
    cam = HAND @ rot.T + np.array([0.12, 0.25, 0.35])  # 35 cm in front, lower right (OpenCV axes)
    px = project(cam, 1920, 1080, 150.0)
    got = fit_hand(HAND, rays(px, 1920, 1080, 150.0))
    assert got is not None
    np.testing.assert_allclose(got[0], cam, atol=1e-6)
    assert got[1] < 1e-3
    # a different (wrong) lens assumption keeps the direction but changes the distance
    wrong = fit_hand(HAND, rays(px, 1920, 1080, 120.0))
    assert wrong is not None and np.linalg.norm(wrong[0][0]) != np.linalg.norm(cam[0])


def test_head_axes_and_wrist_frame():
    head = to_head(np.array([[0.1, 0.2, 0.5]]))  # right 10 cm, 20 cm below, 50 cm ahead
    np.testing.assert_allclose(head, [[0.1, 0.5, -0.2]])
    ax = wrist_axes(to_head(HAND))
    np.testing.assert_allclose(ax @ ax.T, np.eye(3), atol=1e-9)


def test_smoothing_keeps_shape_and_measures_speed():
    frames = []
    for i in range(6):
        wrist = [0.0, 0.4 + 0.05 * i + (0.08 if i == 3 else 0.0), -0.2]  # moves 5 cm per frame, one jitter spike
        kp = [[w + 0.01 * j for w in wrist] for j in range(21)]
        frames.append({"t": i * 0.1, "hands": [{"side": "left", "kp": kp, "wrist": wrist}]})
    smooth_and_speed(frames, Hands3DConfig(smooth_window=3))
    h3 = frames[3]["hands"][0]
    assert abs(h3["wrist"][1] - 0.6) < 1e-9 and abs(h3["raw_wrist"][1] - 0.63) < 1e-9  # spike removed
    assert abs(h3["kp"][5][0] - h3["wrist"][0] - 0.05) < 1e-9  # shape kept (whole hand shifted)
    clean = [{"t": i * 0.1, "hands": [{"side": "right", "kp": [[0.0, 0.05 * i, 0.0]] * 21, "wrist": [0.0, 0.05 * i, 0.0]}]}
             for i in range(5)]
    clean.insert(3, {"t": 0.25, "hands": []})  # a frame without hands does not break the sequence
    smooth_and_speed(clean, Hands3DConfig(smooth_window=3))
    assert [round(f["hands"][0]["speed_mps"], 6) for f in clean if f["hands"]] == [0.5] * 5  # 5 cm per 0.1 s
