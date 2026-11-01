from __future__ import annotations

from pathlib import Path

from egovision.review import Decision, append, apply_review, load_log, queue, reviewed_export
from egovision.schema import Action


def acts():
    return [Action(id="a000", clip="c", start_s=0, end_s=4, verb="wipe", object="slab", hand="right", status="verified"),
            Action(id="a001", clip="c", start_s=4, end_s=6, verb="hold", object="slab", hand="right", status="unverified",
                   flags=["too_much_motion_for_verb"]),
            Action(id="a002", clip="c", start_s=6, end_s=9, verb="push", object="slab", hand="right", status="rejected")]


def test_queue_puts_uncertain_and_unreviewed_first():
    log = [Decision(action_id="a002", op="reject")]
    assert [a.id for a in queue(acts(), log)] == ["a001", "a000", "a002"]


def test_layer_is_persisted_and_replayed(tmp_path: Path):
    append(tmp_path, Decision(action_id="a000", op="approve", by="khushi"))
    append(tmp_path, Decision(action_id="a001", op="edit", fields={"verb": "wipe", "end_s": 6.5, "bogus": 1}))
    append(tmp_path, Decision(action_id="a002", op="reject"))
    log = load_log(tmp_path)
    out = {a.id: a for a in apply_review(acts(), log)}
    assert out["a000"].review["status"] == "approved" and out["a000"].review["by"] == "khushi"
    assert out["a001"].verb == "wipe" and out["a001"].end_s == 6.5 and out["a001"].review["changed"] == ["end_s", "verb"]
    assert [a.id for a in reviewed_export(list(out.values()))] == ["a000", "a001"]
    assert not (tmp_path / ".review.json.tmp").exists()


def test_split_and_merge():
    log = [Decision(action_id="a000", op="split", at_s=2.0), Decision(action_id="a001", op="merge")]
    out = apply_review(acts(), log)
    assert [(a.id, a.start_s, a.end_s) for a in out] == [("a000", 0, 2.0), ("a000b", 2.0, 4), ("a001", 4, 9)]


def test_trim_approval_is_bound_to_the_plan_it_approved():
    from egovision.review import approved_plan, progress, trim_approved

    log = [Decision(op="approve_trim", fields={"plan_sha": "aaa"})]
    assert trim_approved(log, "aaa") and not trim_approved(log, "bbb")  # plan changed after approval
    assert progress(acts(), log, "bbb")["trim_approved"] is False
    assert approved_plan([Decision(op="approve_trim")]) == ""  # approvals stored before fingerprints never match
    assert approved_plan([]) is None
