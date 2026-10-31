from __future__ import annotations

from pathlib import Path

from egovision.evaluate import edit_score, eval_hands_judged, eval_steps, object_match, segmental_counts
from egovision.schema import Action


def A(s, e, verb, obj="x", hand="right"):  # noqa: N802 - short test helper
    return Action(clip="c", start_s=s, end_s=e, verb=verb, object=obj, hand=hand)


def test_perfect_prediction_scores_100():
    gt = [A(0, 2, "pick"), A(2, 5, "wipe", "cloth"), A(1, 4, "hold", hand="left")]
    r = eval_steps(gt, gt, duration=6)
    assert r["frame_acc"] == 100.0 and r["F1@50"] == 100.0 and r["edit"] == 100.0
    assert r["verb_acc_matched"] == 100.0 and r["hand_acc_matched"] == 100.0 and r["boundary_mae_s"] == 0.0


def test_wrong_verb_and_shift():
    gt = [A(0, 4, "wipe", "cloth")]
    pred = [A(1, 4, "push", "cloth")]
    r = eval_steps(pred, gt, duration=4)
    assert r["F1@10"] == 0.0 and r["verb_acc_matched"] == 0.0 and r["object_acc_matched"] == 100.0
    assert r["boundary_mae_s"] == 0.5
    assert r["frame_acc"] == 0.0  # 1 s of pred-idle/GT-wipe + 3 s of push vs wipe


def test_segmental_counts_iou_threshold():
    assert segmental_counts([A(0, 10, "pick")], [A(5, 10, "pick")], 0.5) == (1, 0, 0)
    assert segmental_counts([A(0, 10, "pick")], [A(5, 10, "pick")], 0.6) == (0, 1, 1)


def test_edit_and_object_match():
    assert edit_score(["a", "b", "c"], ["a", "c"]) == (1 - 1 / 3) * 100
    assert object_match("aluminium profile", "profile") and not object_match("cloth", "granite slab")


def test_hands_judged(tmp_path: Path):
    p = tmp_path / "hands_judged.csv"
    p.write_text("clip,frame,gt_hands,correct,false_pos,side_errors,kp_bad,notes\n"
                 "MTL-SKL00089-007,10,2,2,0,0,0,\nmytronlabs-000060-016,20,2,1,1,1,0,blur\n"
                 "mytronlabs-000060-016,30,,,,,,not judged yet\n", encoding="utf-8")
    r = eval_hands_judged(p)
    assert r["all"] == {"frames": 2, "gt_hands": 4, "recall": 75.0, "precision": 75.0, "side_acc": 66.7, "kp_ok": 100.0}
    assert set(r["by_source"]) == {"MTL-SKL00089", "mytronlabs-000060"}
