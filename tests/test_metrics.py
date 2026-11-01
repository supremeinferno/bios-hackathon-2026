"""Metric definitions (egovision.metrics) on hand-computed toy cases."""

from __future__ import annotations

from egovision.evaluate import eval_steps, eval_verification
from egovision.metrics import (
    IDLE,
    MISSED,
    SPURIOUS,
    classification_from_confusion,
    confusion,
    eval_hands_gt,
    frame_pairs,
    hand_diagnostics,
    match_segments,
    segment_confusion,
    verification_counts,
    verification_rates,
)
from egovision.schema import Action, FrameHands, Hand


def A(s, e, verb, obj="x", hand="right", status=None):  # noqa: N802 - short test helper
    return Action(clip="c", start_s=s, end_s=e, verb=verb, object=obj, hand=hand, status=status)


def H(x, y, side="right", w=100, source="detected", hid=None):  # noqa: N802
    kp = [[x + (i % 5) * w / 5, y + (i // 5) * w / 5] for i in range(21)]
    return Hand(id=hid, side=side, score=0.9, bbox=[x, y, x + w, y + w], kp2d=kp, source=source,
                interp=source == "interpolated")


def test_frame_classification_counts_idle_but_does_not_average_it():
    gt = [A(0, 1, "pick")]
    pred = [A(0, 0.5, "pick"), A(0.5, 1, "place")]
    cm = confusion(frame_pairs(pred, gt, 2.0))
    # right lane: 5 pick-pick, 5 pick-place, 10 idle-idle; left lane: 20 idle-idle
    assert cm == {"pick": {"pick": 5, "place": 5}, IDLE: {IDLE: 30}}
    r = classification_from_confusion(cm)
    assert r["accuracy_all"] == 87.5 and r["accuracy_active"] == 50.0
    assert r["per_class"]["pick"]["recall"] == 50.0 and r["per_class"]["pick"]["precision"] == 100.0
    assert r["per_class"]["place"]["f1"] == 0.0
    # macro over pick (66.7) and place (0); weighted by GT support: only pick has support
    assert r["macro_f1"] == round((66.7 + 0.0) / 2, 1) and r["weighted_f1"] == 66.7


def test_both_hand_actions_fill_both_lanes():
    cm = confusion(frame_pairs([A(0, 1, "hold", hand="both")], [A(0, 1, "hold", hand="left")], 1.0))
    assert cm == {"hold": {"hold": 10}, IDLE: {"hold": 10}}


def test_one_to_one_matching_does_not_reuse_a_prediction():
    gt = [A(0, 4, "wipe"), A(4, 8, "wipe")]
    pred = [A(0, 8, "wipe")]
    m = match_segments(pred, gt)
    assert len(m) == 1  # the long prediction can explain only one GT segment
    cm = segment_confusion(pred, gt)
    assert cm["wipe"] == {"wipe": 1, MISSED: 1}


def test_segment_confusion_spurious_row():
    cm = segment_confusion([A(0, 1, "pick"), A(10, 11, "cut")], [A(0, 1, "place")])
    assert cm == {"place": {"pick": 1}, SPURIOUS: {"cut": 1}}


def test_eval_steps_detail_hand_agnostic_and_objects():
    gt = [A(0, 4, "hold", "granite slab", hand="both")]
    pred = [A(0, 4, "hold", "stone slab", hand="right")]
    r = eval_steps(pred, gt, 4.0)
    # per lane: right TP, left FN -> F1 66.7; ignoring hands the segment is found -> 100
    assert r["F1@50"] == 66.7 and r["detail"]["F1@50_any_hand"] == 100.0
    m = r["detail"]["matched_1to1"]
    assert m["hand_acc"] == 0.0 and m["object_exact"] == 0.0 and m["object_lenient"] == 100.0


def test_verification_rates():
    gt = [A(0, 2, "pick"), A(5, 7, "wipe")]
    pred = [A(0, 2, "pick", status="rejected"), A(5, 7, "push", status="verified"),
            A(10, 12, "hold", status="verified"), A(20, 22, "hold", status="rejected")]
    c = verification_counts(pred, gt)
    assert c == {"rejected": {"supported": 1, "unsupported": 1}, "verified": {"supported": 1, "unsupported": 1}}
    r = verification_rates(c)
    assert r["false_rejection_rate"] == 50.0 and r["false_acceptance_rate"] == 50.0 and r["rejection_precision"] == 50.0
    v = eval_verification(pred, gt)
    # with the verb required, the 'push' over a GT 'wipe' is unsupported
    assert v["rates_same_verb"]["false_acceptance_rate"] == round(100 * 2 / 3, 1)


def test_eval_hands_gt_detection_side_landmarks_and_id_switch():
    pred = [
        FrameHands(frame=0, t=0, hands=[H(0, 0, "left", hid=1), H(500, 0, "right", hid=2)]),
        FrameHands(frame=2, t=0.07, hands=[H(0, 0, "right", hid=3)]),  # wrong side and a new id
        FrameHands(frame=4, t=0.13, hands=[H(800, 800, "right", hid=4, source="interpolated")]),
    ]
    g_left = {"id": "L", "side": "left", "bbox": [0, 0, 100, 100], "kp2d": H(0, 0).kp2d}
    gt = [{"frame": 0, "hands": [g_left]}, {"frame": 2, "hands": [g_left]},
          {"frame": 4, "hands": [{"id": "R", "side": "right", "bbox": [0, 0, 100, 100]}]}]
    r = eval_hands_gt(pred, gt)
    assert r["detection"]["tp"] == 2 and r["detection"]["fp"] == 2 and r["detection"]["fn"] == 1
    assert r["handedness_acc"] == 50.0 and r["id_switches"] == 1
    assert r["landmarks_by_source"]["detected"]["mean_px"] == 0.0
    assert "interpolated" not in r["landmarks_by_source"]  # never pooled with detected landmarks
    assert r["precision_by_source"]["interpolated"] == 0.0


def test_hand_diagnostics_flags_duplicate_sides_and_ignores_interpolated():
    frames = [
        FrameHands(frame=0, t=0.0, hands=[H(0, 0, "right", hid=1), H(600, 0, "right", hid=2)]),
        FrameHands(frame=1, t=0.033, hands=[H(0, 0, "right", source="interpolated", hid=1)]),
        FrameHands(frame=2, t=0.067, hands=[]),
    ]
    d = hand_diagnostics(frames, fps=30, width=1000, stride=2)
    assert d["frames"] == 2 and d["with_hand_pct"] == 50.0 and d["same_side_duplicate_pct"] == 50.0
    assert d["track_ids"] == 2
