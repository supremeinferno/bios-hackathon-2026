from __future__ import annotations

import pytest
from pydantic import ValidationError

from egovision.config import DEFAULT_VERBS, Config
from egovision.schema import ACTIONS_CSV_COLUMNS, Action, Hand


def test_defaults_and_overrides():
    cfg = Config.load(None, ["ingest.stride=3", "hands.swap_handedness=false", "actions.backend=offline"])
    assert cfg.ingest.stride == 3
    assert cfg.hands.swap_handedness is False
    assert cfg.actions.backend == "offline"
    assert cfg.actions.verbs == DEFAULT_VERBS and "other" in cfg.actions.verbs


def test_unknown_config_key_rejected():
    with pytest.raises(ValidationError):
        Config.load(None, ["ingest.strid=3"])
    with pytest.raises(ValueError):
        Config.load(None, ["no_equals_sign"])


def test_hand_validation():
    kp = [[0.0, 0.0]] * 21
    Hand(side="left", score=0.5, bbox=[0, 0, 1, 1], kp2d=kp)
    with pytest.raises(ValidationError):
        Hand(side="left", score=0.5, bbox=[5, 0, 1, 1], kp2d=kp)  # x2 < x1
    with pytest.raises(ValidationError):
        Hand(side="left", score=0.5, bbox=[0, 0, 1, 1], kp2d=kp[:20])
    with pytest.raises(ValidationError):
        Hand(side="middle", score=0.5, bbox=[0, 0, 1, 1], kp2d=kp)  # type: ignore[arg-type]


def test_action_schema_and_csv_row():
    a = Action(clip="c", start_s=1.234, end_s=2.5, verb="pick", object="screw", hand="right")
    assert a.csv_row() == ["c", "1.23", "2.50", "pick", "screw", "right"]
    assert ACTIONS_CSV_COLUMNS == ["clip", "start_s", "end_s", "verb", "object", "hand"]  # PDF p.4
    with pytest.raises(ValidationError):
        Action(clip="c", start_s=3, end_s=2, verb="pick", object="x", hand="left")
