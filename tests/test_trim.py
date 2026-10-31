from __future__ import annotations

from pathlib import Path

import av

from egovision.config import TrimConfig
from egovision.schema import Action
from egovision.trim import plan, write_trimmed


def A(s, e, status=None, review=None):  # noqa: N802
    return Action(clip="c", start_s=s, end_s=e, verb="wipe", object="x", hand="right", status=status, review=review)


def test_plan_pads_merges_and_skips_rejected():
    cfg = TrimConfig(pad_s=1.0, merge_gap_s=2.0)
    segs = plan([A(5, 6), A(8, 9), A(20, 21, status="rejected"), A(30, 31, review={"status": "rejected"})], 40, cfg)
    assert segs == [{"start_s": 0.0, "end_s": 4.0, "mode": "lite"}, {"start_s": 4.0, "end_s": 10.0, "mode": "keep"},
                    {"start_s": 10.0, "end_s": 40.0, "mode": "lite"}]


def test_keep_lite_preserves_timestamps(tmp_path: Path, synthetic_video: Path):
    cfg = TrimConfig(lite_fps=2.0, width=160, codec="libx264", out_dir=tmp_path)
    segs = [{"start_s": 0.0, "end_s": 1.0, "mode": "keep"}, {"start_s": 1.0, "end_s": 2.0, "mode": "lite"}]
    out = tmp_path / "t.mp4"
    n = write_trimmed(synthetic_video, out, segs, 10.0, cfg)
    assert n == {"keep_frames": 10, "lite_frames": 2, "lite_every_n": 5}
    with av.open(str(out)) as c:
        times = [round(float(f.pts * f.time_base), 2) for f in c.decode(video=0)]
    assert times[:10] == [i / 10 for i in range(10)] and times[10:] == [1.0, 1.5]  # original timestamps kept
