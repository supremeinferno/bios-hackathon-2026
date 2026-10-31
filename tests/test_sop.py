from __future__ import annotations

from egovision.config import SOPConfig
from egovision.schema import Action
from egovision.sop import check, discover, lcs, sequence


def acts(*steps, status=None):
    out = []
    for i, s in enumerate(steps):
        v, o = s.split(":")
        out.append(Action(clip="c", start_s=10 * i, end_s=10 * i + 5, verb=v, object=o, hand="right", status=status))
    return out


CFG = SOPConfig()


def test_lcs_monotone():
    assert lcs(["a", "c", "b"], ["a", "b", "c"]) in ([(0, 0), (1, 2)], [(0, 0), (2, 1)])


def test_discover_and_check():
    clips = {
        "x-1": (acts("pick:cloth", "wipe:slab", "place:cloth"), 30.0),
        "x-2": (acts("pick:cloth", "wipe:slab", "place:cloth", "other:thing"), 40.0),
        "x-3": (acts("pick:cloth", "wipe:slab"), 20.0),
    }
    t = discover(clips, CFG)
    assert [s["key"] for s in t] == ["pick:cloth", "wipe:slab", "place:cloth"] and t[0]["step_id"] == "S1"
    ok = check(acts("pick:cloth", "wipe:slab", "place:cloth"), t, CFG)
    assert ok["missing"] == [] and ok["completeness"] == 1.0
    bad = check(acts("wipe:slab", "pick:cloth", "wipe:slab"), t, CFG)
    assert "S3" in bad["missing"] and bad["repeated"] == ["wipe:slab"] and bad["out_of_order"]


def test_sequence_drops_rejected_and_collapses_repeats():
    a = acts("pick:cloth", "pick:cloth", "wipe:slab")
    a[2] = a[2].model_copy(update={"status": "rejected"})
    assert sequence(a, CFG) == ["pick:cloth"]


def test_canonical_object_groups_wording_variants():
    from egovision.steps.postprocess import canonical_object

    same = ["wooden piece", "wood piece", "piece of wood", "wooden pieces", "Wood Pieces", "the wooden piece"]
    assert {canonical_object(o) for o in same} == {"wood"}
    assert canonical_object("wood piece to furniture frame") == "wood"
    assert canonical_object("wooden frame") == "wood frame" and canonical_object("piece of plywood") == "plywood"
    assert canonical_object("glass") == "glass"  # 'ss' plural guard
