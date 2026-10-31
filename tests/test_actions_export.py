from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from egovision.config import DEFAULT_VERBS, ObjectsConfig
from egovision.export import read_actions_csv, read_hands_jsonl, write_actions_csv, write_hands_jsonl
from egovision.objects.contact import associate, contact_segments
from egovision.schema import Action, Contact, ContactSegment, FrameHands, FrameObjects, ObjectDet
from egovision.steps.gemini import _parse_json_array, build_prompt
from egovision.steps.postprocess import clean, from_raw, normalise_verb, parse_time
from egovision.steps.temporal import carve_pick_place, drop_short, merge_both, runs, viterbi

from .conftest import make_hand


def test_parse_time_formats():
    assert parse_time(3) == 3.0
    assert parse_time("12.5") == 12.5
    assert parse_time("01:05") == 65.0
    assert parse_time("0:01:05.5") == 65.5
    with pytest.raises(ValueError):
        parse_time("soon")


def test_verb_normalisation():
    assert normalise_verb("Pick", DEFAULT_VERBS) == "pick"
    assert normalise_verb("picking up", DEFAULT_VERBS) == "pick"
    assert normalise_verb("juggle", DEFAULT_VERBS) == "other"


def test_from_raw_repairs_and_skips():
    raw = [
        {"start_s": "00:01", "end_s": 2.5, "verb": "Fasten", "object": "The Screw", "hand": "R", "confidence": 0.8},
        {"start_s": 5, "end_s": 4, "verb": "pick", "object": "x", "hand": "left"},  # end before start
        {"start": "bad", "end": 3},  # unparseable
    ]
    acts = from_raw(raw, "clip", DEFAULT_VERBS, "test")
    assert len(acts) == 1
    a = acts[0]
    assert (a.start_s, a.end_s, a.verb, a.object, a.hand, a.confidence) == (1.0, 2.5, "fasten", "screw", "right", 0.8)


def test_clean_clips_and_sorts():
    acts = [Action(clip="c", start_s=8, end_s=12, verb="pick", object="x", hand="left"),
            Action(clip="c", start_s=1, end_s=1.1, verb="pick", object="x", hand="left"),
            Action(clip="c", start_s=0.5, end_s=3, verb="place", object="x", hand="right")]
    out = clean(acts, duration_s=10)
    assert [(a.start_s, a.end_s) for a in out] == [(0.5, 3.0), (8.0, 10.0)]


def test_gemini_helpers():
    assert _parse_json_array('```json\n[{"verb": "pick"}]\n```') == [{"verb": "pick"}]
    with pytest.raises(ValueError):
        _parse_json_array("no array here")
    p = build_prompt(DEFAULT_VERBS, 180)
    assert all(f'"{v}"' in p for v in DEFAULT_VERBS)


def test_vocab_for_prefix_and_fallback():
    cfg = ObjectsConfig()
    assert any(v.name == "end cap" for v in cfg.vocab_for("MTL-SKL00089-007"))
    union = {v.name for v in cfg.vocab_for("factory_001")}
    assert {"end cap", "cloth", "jigsaw"} <= union


def test_contact_rule_prefers_portable():
    cfg = ObjectsConfig()
    hand = make_hand("right", x=100, y=100, hid=1)  # kp i at (100+i, 100+i); tips at 104..120
    tool = ObjectDet(label="screwdriver", kind="portable", score=0.5, bbox=[95, 95, 125, 125])
    table = ObjectDet(label="workbench", kind="fixed", score=0.9, bbox=[0, 0, 500, 500])
    far = ObjectDet(label="pen", kind="portable", score=0.9, bbox=[400, 400, 450, 450])
    (c,) = associate([hand], [table, tool, far], cfg)
    assert (c.state, c.object, c.n_tips) == ("portable", "screwdriver", 5)
    (c2,) = associate([hand], [far], cfg)
    assert c2.state == "none" and c2.object is None


def test_contact_segments_drop_noise():
    def fo(i: float, state: str, obj: str | None) -> FrameObjects:
        return FrameObjects(frame=int(i * 30), t=i, contacts=[Contact(hand_id=1, side="right", state=state, object=obj)])  # type: ignore[arg-type]

    seq = [fo(0.5 * k, "portable", "cloth") for k in range(6)] + [fo(3.0, "none", None)] + \
          [fo(3.5 + 0.5 * k, "portable", "cloth") for k in range(4)]
    segs = contact_segments(seq, "c", step_s=0.5, min_s=0.6)
    assert len(segs) == 1 and segs[0].object_name == "cloth" and segs[0].t_start == 0.0 and segs[0].t_end == 5.25


def test_viterbi_switch_penalty_smooths():
    em = np.log(np.array([[0.9, 0.1], [0.9, 0.1], [0.4, 0.6], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]))
    assert viterbi(em, 0.0).tolist() == [0, 0, 1, 0, 1, 1, 1]
    assert viterbi(em, 2.0).tolist() == [0, 0, 0, 0, 1, 1, 1]
    rs = runs(np.array([2, 2, 0, 2, 1, 1]))
    assert rs == [(2, 0, 1), (0, 2, 2), (2, 3, 3), (1, 4, 5)]
    times = np.arange(6) * 0.5
    assert drop_short(rs, times, 0.5, 0.9, idle=2) == [(2, 0, 3), (1, 4, 5)]


def test_carve_pick_place_and_merge_both():
    hold = Action(clip="c", start_s=1.0, end_s=6.0, verb="hold", object="cloth", hand="right")
    c = ContactSegment(clip="c", t_start=1.0, t_end=6.0, hand="right", state="portable", object_name="cloth")
    out = sorted(carve_pick_place([hold], [c], 0.8, 0.5), key=lambda a: a.start_s)
    assert [(a.verb, a.start_s, a.end_s) for a in out] == [("pick", 1.0, 1.8), ("hold", 1.8, 5.2), ("place", 5.2, 6.0)]
    left = Action(clip="c", start_s=0, end_s=4, verb="wipe", object="granite slab", hand="left")
    right = Action(clip="c", start_s=0.5, end_s=4.2, verb="wipe", object="granite slab", hand="right")
    other = Action(clip="c", start_s=5, end_s=6, verb="push", object="granite slab", hand="right")
    merged = merge_both([left, right, other], 0.5)
    assert [(a.hand, a.start_s, a.end_s) for a in merged] == [("both", 0, 4.2), ("right", 5, 6)]


def test_export_roundtrip(tmp_path: Path):
    frames = [FrameHands(frame=0, t=0, hands=[make_hand("left", hid=0)]), FrameHands(frame=1, t=0.033, hands=[])]
    write_hands_jsonl(frames, tmp_path / "hands.jsonl")
    back = read_hands_jsonl(tmp_path / "hands.jsonl")
    assert [len(f.hands) for f in back] == [1, 0] and back[0].hands[0].side == "left"
    acts = [Action(clip="c", start_s=0, end_s=1, verb="pick", object="x", hand="left")]
    write_actions_csv(acts, tmp_path / "actions.csv")
    rows = list(csv.reader(open(tmp_path / "actions.csv", encoding="utf-8")))
    assert rows[0] == ["clip", "start_s", "end_s", "verb", "object", "hand"]
    assert rows[1] == ["c", "0.00", "1.00", "pick", "x", "left"]
    assert read_actions_csv(tmp_path / "actions.csv")[0].verb == "pick"
