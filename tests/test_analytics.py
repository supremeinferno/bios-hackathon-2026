from __future__ import annotations

from egovision.analytics import clip_analytics
from egovision.schema import Action


def test_analytics_durations_busy_cycles_and_placements():
    acts = [Action(clip="c", start_s=s, end_s=s + 2, verb="wipe", object="slab", hand="right") for s in (0, 4, 8, 16)]
    acts.append(Action(clip="c", start_s=2, end_s=3, verb="place", object="cloth", hand="left"))
    rows = [{"t": i / 10, "left_visible": 1, "left_x": -0.5, "left_y": -0.5, "left_zone": "lower-left",
             "right_visible": 1 if i < 100 else 0, "right_x": 0.5 if i < 100 else "", "right_y": -0.5 if i < 100 else "",
             "right_zone": "lower-right" if i < 100 else ""} for i in range(200)]
    a = clip_analytics(acts, rows, 20.0)
    assert a["time_per_verb_s"]["wipe"] == {"total": 8.0, "count": 4, "mean": 2.0}
    assert a["busy_share"]["right"] == 0.4 and a["two_handed_share"] == 0.0
    assert a["idle"]["right"]["not_visible_share"] == 0.5
    assert a["zone_share"]["left"] == {"lower-left": 1.0}
    assert a["placements"][0]["zone"] == "lower-left" and a["placements"][0]["object"] == "cloth"
    assert a["cycles"]["step"] == "wipe:slab" and a["cycles"]["median_cycle_s"] == 4.0
    assert a["cycles"]["slow_cycles"] == [8.0]
