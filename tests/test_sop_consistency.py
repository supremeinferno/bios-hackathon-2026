"""Cycle consistency, labour cost and automation candidate indicator (egovision/sop_cycles.py), synthetic actions.

A synthetic cycle is 10 s: pick:a 0-2 s, place:a 2.5-4.5 s, push:b 5-7 s, then 3 s with nothing labelled."""

from __future__ import annotations

import json

import pytest

from egovision.config import SOPCyclesConfig, WageConfig
from egovision.schema import Action
from egovision.sop_cycles import ClipData, analyse_source, levenshtein, order_distance, steps_csv

STEPS = [("pick", "a", 0.0, 2.0), ("place", "a", 2.5, 4.5), ("push", "b", 5.0, 7.0)]


def act(verb: str, obj: str, s: float, e: float, status: str = "verified", clip: str = "S-001") -> Action:
    return Action(clip=clip, start_s=s, end_s=e, verb=verb, object=obj, hand="right", status=status)  # type: ignore[arg-type]


def cycles(n: int, t0: float = 0.0, steps: list[tuple[str, str, float, float]] | None = None) -> list[Action]:
    return [act(v, o, t0 + 10 * i + s, t0 + 10 * i + e) for i in range(n) for v, o, s, e in (steps or STEPS)]


def closing_anchor(t: float) -> list[Action]:
    return [act("pick", "a", t, t + 2)]  # next cycle starts but the clip ends: a partial cycle


def run(actions: list[Action], **cfg: object) -> dict:
    c = SOPCyclesConfig(min_anchor_clips=1, **cfg)  # type: ignore[arg-type]
    return analyse_source("S", {"S-001": ClipData(actions, 60.0, "lab")}, c)


def test_levenshtein_and_distance():
    assert levenshtein(list("abc"), list("acb")) == 2
    assert order_distance(["x", "y"], ["x", "y"]) == 0
    assert order_distance(["x"], ["y", "z"]) == 1


def test_identical_cycles_score_100():
    rep = run(cycles(5) + closing_anchor(50))
    g = rep["groups"][0]
    assert g["status"] == "ok" and g["anchor"]["step"] == "pick:a" and g["complete_cycles"] == 5
    assert [t["key"] for t in g["template"]] == ["pick:a", "place:a", "push:b"]
    assert g["order_distance"] == {"mean": 0.0, "std": 0.0}
    assert g["cycle_time"]["cv"] == 0.0 and g["consistency_score"] == 100.0
    assert g["labour"]["idle_share"] == pytest.approx(0.4)  # 4 s of 10 s labelled with nothing
    steps = {s["step"]: s for s in g["steps"]}
    assert all(s["cv"] == 0.0 and s["n_cycles"] == 5 for s in steps.values())
    assert steps["pick:a"]["indicator"] == "candidate" and steps["push:b"]["indicator"] == "candidate"
    assert g["labour"]["cost_per_cycle_median"] is None and rep["wage"]["declared"] is False  # never inferred


def test_one_swapped_step_gives_expected_distance():
    swapped = [("pick", "a", 0.0, 2.0), ("push", "b", 2.5, 4.5), ("place", "a", 5.0, 7.0)]
    acts = cycles(2) + cycles(1, 20, swapped) + cycles(2, 30) + closing_anchor(50)
    g = run(acts)["groups"][0]
    d = [c["order_distance"] for c in g["cycles"]]
    assert d == [0, 0, round(2 / 3, 3), 0, 0]  # two substitutions over three steps
    assert g["order_distance"]["mean"] == pytest.approx(2 / 3 / 5, abs=1e-3)
    assert g["consistency_score"] == pytest.approx(100 * (0.5 * (1 - 2 / 15) + 0.5), abs=0.1)


def test_partial_cycles_at_clip_edges_are_excluded():
    head = [act("place", "a", 2.5, 4.5), act("push", "b", 5.0, 7.0)]  # clip starts mid-cycle
    tail = [act("pick", "a", 40, 42), act("place", "a", 42.5, 44.5)]  # clip ends mid-cycle
    rep = run(head + cycles(3, 10) + tail, anchors={"S": "pick:a"})
    g = rep["groups"][0]
    assert g["complete_cycles"] == 3 and [c["start_s"] for c in g["cycles"]] == [10, 20, 30]
    assert all(c["start_s"] >= 10 and c["end_s"] <= 40 for c in g["cycles"])
    assert g["anchor"]["chosen_by"] == "config"


def test_fewer_than_three_cycles_is_insufficient_data():
    g = run(cycles(2) + closing_anchor(20))["groups"][0]
    assert g["complete_cycles"] == 2 and g["status"] == "insufficient data"
    assert "consistency_score" not in g and "steps" not in g


def test_unverified_actions_are_excluded_and_counted():
    acts = cycles(5) + closing_anchor(50)
    acts[2] = acts[2].model_copy(update={"status": "unverified"})  # push:b of cycle 1
    acts.append(act("wipe", "c", 7.5, 8.5, status="rejected"))
    rep = run(acts)
    assert rep["actions"] == {"total": 17, "used": 15, "excluded": 2, "excluded_share": round(2 / 17, 3)}
    g = rep["groups"][0]
    push = next(s for s in g["steps"] if s["step"] == "push:b")
    assert push["n_cycles"] == 4 and push["presence"] == 0.8
    # unverified time is reported apart, not as idle; the rejected wipe is not evidence of activity
    assert g["cycles"][0]["unverified_s"] == 2.0 and g["cycles"][0]["idle_s"] == 4.0


def test_wage_labeller_filter_and_wording():
    other = ClipData(cycles(5) + closing_anchor(50), 60.0, "other-labeller")
    c = SOPCyclesConfig(min_anchor_clips=1, wage=WageConfig(currency="INR", rate_per_hour=360))
    rep = analyse_source("S", {"S-001": ClipData(cycles(5) + closing_anchor(50), 60.0, "lab"),
                               "S-002": ClipData(cycles(5) + closing_anchor(50), 60.0, "lab"), "S-003": other}, c)
    assert rep["clips_excluded"] == {"S-003": "labeller other-labeller (majority: lab)"}
    g = rep["groups"][0]
    assert g["labour"]["cost_per_cycle_median"] == pytest.approx(1.0)  # 10 s at 360 INR/h
    rows = steps_csv(rep.pop("_step_rows"), rep["wage"])
    assert rows.splitlines()[0].startswith("group,step,n_cycles") and ",INR," in rows
    assert "worker" not in (json.dumps(rep) + rows).lower()


def test_label_noise_flag_when_the_same_window_has_another_label():
    acts = cycles(5) + closing_anchor(50)
    acts += [act("hold", "b", 10 * i + 5.0, 10 * i + 7.0, status="unverified") for i in range(3)]  # same hand, same time
    steps = {s["step"]: s for s in run(acts)["groups"][0]["steps"]}
    assert any(f.startswith("label_noise (3/5") for f in steps["push:b"]["flags"])
    assert not any(f.startswith("label_noise") for f in steps["pick:a"]["flags"])
