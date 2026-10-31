from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from egovision.config import PrivacyConfig
from egovision.ingest import iter_all_frames, probe
from egovision.privacy import Sanitizer, frame_mask, is_wearer_body, pixelate, regions_from_detections
from egovision.schema import FrameHands, ObjectDet

from .conftest import make_hand

W, H = 1920, 1080
CFG = PrivacyConfig(pad_frac=0.0)


def det(cat: str, box: list[float], score: float = 0.5) -> ObjectDet:
    return ObjectDet(label=f"privacy:{cat}", kind="privacy", score=score, bbox=box)


def test_person_becomes_head_band_and_objects_are_ignored():
    regs = regions_from_detections([det("person", [100, 100, 300, 500]), det("sign", [10, 10, 50, 50]),
                                    ObjectDet(label="cloth", kind="portable", score=0.9, bbox=[0, 0, 5, 5])],
                                   [], W, H, CFG)
    assert [(r.category, r.box) for r in regs] == [("head", [100, 100, 300, 240.0]), ("sign", [10, 10, 50, 50])]


def test_wearer_arm_is_not_blurred():
    hand = make_hand("right", x=900, y=950)  # bbox [900, 950, 920, 970]
    arm = [850, 600, 1100, 1080]  # touches the bottom edge and covers the wearer's hand
    coworker = [1500, 100, 1700, 600]
    assert is_wearer_body(arm, [hand], W, H) and not is_wearer_body(coworker, [hand], W, H)
    assert is_wearer_body([0, 300, 1920, 1080], [], W, H)  # huge, anchored at the bottom = the wearer
    regs = regions_from_detections([det("person", arm), det("person", coworker)], [hand], W, H, CFG)
    assert len(regs) == 1 and regs[0].box[0] == 1500


def test_hold_window():
    s = Sanitizer.__new__(Sanitizer)
    s.cfg, s.keys = PrivacyConfig(hold_s=0.5), []
    s.meta = None
    from egovision.privacy import KeyRegions, Region

    s.keys = [KeyRegions(0, 1.0, [Region("face", [0, 0, 1, 1], 0.5, "privacy:face")]), KeyRegions(30, 2.0, [])]
    assert len(s.active(0.6)) == 1 and len(s.active(1.5)) == 1 and s.active(1.6) == []


def test_mask_cuts_out_hands_and_pixelation_changes_only_masked_pixels():
    m = frame_mask((100, 200), [[0, 0, 100, 50]], [[20, 10, 40, 30]], s=1.0)
    assert m[5, 5] == 1 and m[20, 30] == 0 and m[80, 150] == 0
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
    out = np.where(m[..., None].astype(bool), pixelate(img, 10), img)
    assert (out[m == 0] == img[m == 0]).all() and not (out[m == 1] == img[m == 1]).all()


def test_sanitizer_writes_video(tmp_path: Path, synthetic_video: Path):
    meta = probe(synthetic_video)
    s = Sanitizer(meta, PrivacyConfig(output_width=160, pixel_block=8, hold_s=0.2))
    s.add(5, 0.5, [det("face", [20, 20, 120, 100])], [])
    hands = [FrameHands(frame=i, t=i / 10, hands=[]) for i in range(meta.n_frames)]
    out = tmp_path / "san.mp4"
    summary = s.write(hands, out)
    assert out.is_file() and summary["regions_by_category"] == {"face": 1}
    assert summary["frames_blurred_pct"] == pytest.approx(25.0)  # frames 3..7 of 20 (t within 0.5 +/- 0.2)
    orig = dict(iter_all_frames(str(synthetic_video)))
    san = dict(iter_all_frames(str(out)))
    assert len(san) == 20
    diff = np.abs(san[5].astype(int) - orig[5].astype(int))
    assert diff[30:90, 30:110].mean() > diff[105:, 125:].mean()  # changed inside the region, not outside
