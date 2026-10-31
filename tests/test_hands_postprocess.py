from __future__ import annotations

import pytest

from egovision.hands.postprocess import assign_side_ids, fill_gaps
from egovision.schema import FrameHands

from .conftest import make_hand


def test_assign_side_ids_keeps_best_per_side():
    fh = FrameHands(frame=0, t=0, hands=[make_hand("left", score=0.5), make_hand("left", score=0.9),
                                         make_hand("right", score=0.7)])
    out = assign_side_ids([fh])[0]
    assert [(h.side, h.id, h.score) for h in out.hands] == [("left", 0, 0.9), ("right", 1, 0.7)]


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
