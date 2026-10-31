from __future__ import annotations

import csv
from pathlib import Path

import pytest

from egovision.actions.gemini import _parse_json_array, build_prompt
from egovision.actions.offline import HandPresenceBaseline
from egovision.actions.postprocess import clean, from_raw, normalise_verb, parse_time
from egovision.config import DEFAULT_VERBS, ActionsConfig
from egovision.export import read_hands_jsonl, write_actions_csv, write_hands_jsonl
from egovision.schema import Action, FrameHands, VideoMeta

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


def test_parse_json_array_with_fences():
    assert _parse_json_array('```json\n[{"verb": "pick"}]\n```') == [{"verb": "pick"}]
    with pytest.raises(ValueError):
        _parse_json_array("no array here")


def test_prompt_lists_every_verb():
    p = build_prompt(DEFAULT_VERBS, 180)
    assert all(f'"{v}"' in p for v in DEFAULT_VERBS)


def test_offline_baseline_segments():
    meta = VideoMeta(path="x", name="c", width=100, height=100, fps=10, n_frames=40, duration_s=4, codec="x")
    frames = []
    for i in range(40):
        hands = [make_hand("left", hid=0)] if 10 <= i < 30 else []
        if 20 <= i < 30:
            hands.append(make_hand("right", hid=1))
        frames.append(FrameHands(frame=i, t=i / 10, hands=hands))
    acts = HandPresenceBaseline(ActionsConfig(min_action_s=0.5, merge_gap_s=0.2)).segment(meta, frames)
    assert [(a.hand, a.start_s, a.end_s) for a in acts] == [("left", 1.0, 2.0), ("both", 2.0, 3.0)]
    assert all(a.verb == "other" and a.source == "offline-hand-presence" for a in acts)


def test_export_roundtrip(tmp_path: Path):
    frames = [FrameHands(frame=0, t=0, hands=[make_hand("left", hid=0)]), FrameHands(frame=1, t=0.033, hands=[])]
    write_hands_jsonl(frames, tmp_path / "hands.jsonl")
    back = read_hands_jsonl(tmp_path / "hands.jsonl")
    assert [len(f.hands) for f in back] == [1, 0] and back[0].hands[0].side == "left"
    write_actions_csv([Action(clip="c", start_s=0, end_s=1, verb="pick", object="x", hand="left")],
                      tmp_path / "actions.csv")
    rows = list(csv.reader(open(tmp_path / "actions.csv", encoding="utf-8")))
    assert rows[0] == ["clip", "start_s", "end_s", "verb", "object", "hand"]
    assert rows[1] == ["c", "0.00", "1.00", "pick", "x", "left"]
