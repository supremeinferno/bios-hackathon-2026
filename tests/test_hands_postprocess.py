from __future__ import annotations

import pytest

from egovision.config import HandsConfig
from egovision.hands.postprocess import fill_gaps
from egovision.hands.tracking import smooth_tracks, track_hands
from egovision.schema import FrameHands

from .conftest import make_hand

CFG = HandsConfig()


def _fh(i: int, *hands) -> FrameHands:
    return FrameHands(frame=i, t=i / 30, hands=list(hands))


def test_tracking_keeps_ids_and_fixes_side_flicker():
    frames = [_fh(i, make_hand("left", x=100 + i, score=0.9), make_hand("right", x=900 + i, score=0.9))
              for i in range(0, 20, 2)]
    # one-frame L/R flip on the left hand
    frames[4] = _fh(8, make_hand("right", x=108, score=0.6), make_hand("right", x=908, score=0.9))
    out = track_hands(frames, fps=30, width=1920, height=1080, cfg=CFG)
    ids = {(h.side, h.id) for fh in out for h in fh.hands}
    assert ids == {("left", 0), ("right", 1)}  # stable IDs, flip corrected by side vote


def test_reentry_reuses_id_of_same_side():
    frames = [_fh(0, make_hand("left", x=100)), _fh(2, make_hand("left", x=102)),
              _fh(40, make_hand("left", x=600))]  # gone > max_age, back within reid window, far away
    out = track_hands(frames, fps=30, width=1920, height=1080, cfg=CFG)
    assert [fh.hands[0].id for fh in out] == [0, 0, 0]
    late = track_hands([frames[0], _fh(200, make_hand("left", x=600))], 30, 1920, 1080, CFG)
    assert [fh.hands[0].id for fh in late] == [0, 1]  # outside the 3 s window -> new ID


def test_smoothing_reduces_jitter():
    import numpy as np

    frames = [_fh(i, make_hand("left", x=100 + (3 if i % 4 else -3), hid=0)) for i in range(0, 60, 2)]
    sm = smooth_tracks(frames, CFG)
    raw_x = np.array([fh.hands[0].kp2d[0][0] for fh in frames])
    sm_x = np.array([fh.hands[0].kp2d[0][0] for fh in sm])
    assert np.std(np.diff(sm_x)) < 0.5 * np.std(np.diff(raw_x))


def test_fill_gaps_interpolates_same_id():
    a = FrameHands(frame=0, t=0.0, hands=[make_hand("left", x=0, hid=0)])
    b = FrameHands(frame=2, t=0.0667, hands=[make_hand("left", x=10, hid=0)])
    out = fill_gaps([a, b], n_frames=3, fps=30, max_gap=6)
    assert [fh.frame for fh in out] == [0, 1, 2]
    mid = out[1].hands[0]
    assert mid.interp and mid.bbox[0] == pytest.approx(5.0) and mid.kp2d[0][0] == pytest.approx(5.0)
    assert not out[0].hands[0].interp


def test_fill_gaps_never_invents_hands():
    a = FrameHands(frame=0, t=0, hands=[make_hand("left", hid=0)])
    b = FrameHands(frame=2, t=0, hands=[make_hand("right", hid=1)])  # different id on each side
    assert fill_gaps([a, b], 3, 30, 6)[1].hands == []
    c = FrameHands(frame=10, t=0, hands=[make_hand("left", hid=0)])
    assert fill_gaps([a, c], 11, 30, max_gap=6)[5].hands == []  # gap too long
    tail = fill_gaps([a], 4, 30, 6)  # frames after the last processed frame stay empty
    assert len(tail) == 4 and tail[3].hands == []
