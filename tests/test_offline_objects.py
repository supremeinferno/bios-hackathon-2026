"""Offline step labeller: exemplar features folder, nearest-object naming fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from egovision.config import StepsConfig
from egovision.objects import nearest_object
from egovision.schema import ObjectDet, VideoMeta
from egovision.steps.exemplar import ExemplarMatcher
from egovision.steps.features import KeyframeFeatures, SideFeat

from .conftest import make_hand


def Det(label, box, kind="portable", score=0.5):
    return ObjectDet(label=label, kind=kind, score=score, bbox=box)


def test_nearest_object_prefers_close_portable_then_fixed():
    h = make_hand("right", x=100, y=100)  # 20 px hand; fingertips around (112-120, 112-120)
    near, far = Det("cloth", [130, 100, 150, 140]), Det("tape", [400, 400, 420, 420])
    conveyor = Det("roller conveyor", [0, 0, 1000, 1000], kind="fixed")
    assert nearest_object(h, [far, near, conveyor], max_dist=1.5).label == "cloth"
    assert nearest_object(h, [far, conveyor], max_dist=1.5).label == "roller conveyor"  # no portable close enough
    assert nearest_object(h, [far], max_dist=1.5) is None
    assert nearest_object(h, [Det("p", [0, 0, 1, 1], kind="privacy")], max_dist=99) is None


def _matcher(**upd) -> ExemplarMatcher:
    m = ExemplarMatcher.__new__(ExemplarMatcher)
    m.cfg = StepsConfig(**upd)
    m._text, m.visual = {}, None
    return m


def _keyframes(near: str | None) -> list[KeyframeFeatures]:
    out = []
    for k in range(10):
        kf = KeyframeFeatures(frame=k * 10, t=k * 0.33)
        kf.sides = {"right": SideFeat(present=True, hand_id=0, speed=1.0, near_object=near),
                    "left": SideFeat(present=False)}
        out.append(kf)
    return out


META = VideoMeta(path="x.mp4", name="c", width=1920, height=1080, fps=30.0, n_frames=100, duration_s=3.3, codec="h264")


def test_nearest_fallback_names_the_object_only_when_switched_on():
    for fallback, want in (("none", "unknown"), ("nearest", "cloth")):
        m = _matcher(object_fallback=fallback, pick_place_s=0, w_text=0, w_cue=0, idle_prob_empty_moving=0.01)
        acts = m.segment(META, [], _keyframes("cloth"), [])
        assert acts and {a.object for a in acts} == {want}
        if fallback == "nearest":
            assert acts[0].evidence == {"object_source": "nearest"}


def test_exemplars_read_features_from_the_configured_folder(tmp_path: Path):
    d = tmp_path / "runs" / "clipX"
    d.mkdir(parents=True)
    emb = np.zeros((3, 4), np.float16)
    emb[:, 0] = 1
    np.savez(d / "features.npz", t=np.array([1.0, 2.0, 9.0]), left=np.full((3, 4), np.nan, np.float16), right=emb)
    ex = tmp_path / "ex.csv"
    ex.write_text("clip,start_s,end_s,verb,object,hand\nclipX,0.5,2.5,wipe,slab,right\n", encoding="utf-8")
    old = _matcher(exemplars_features_root=tmp_path / "elsewhere")._load_visual_prototypes(ex)
    assert old is None  # wrong folder: no prototypes (and a warning)
    protos = _matcher(exemplars_features_root=tmp_path / "runs")._load_visual_prototypes(ex)
    verbs = StepsConfig().verbs
    assert protos is not None and protos[verbs.index("wipe")][0] == 1.0 and not protos[verbs.index("pick")].any()
