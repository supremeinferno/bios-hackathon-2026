from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from egovision.config import Config, HandsConfig
from egovision.hands.mediapipe_backend import MediaPipeHands
from egovision.hands.rtdetr import accept, crop_to_frame, square_crop
from egovision.ingest import Frame
from egovision.schema import VideoMeta

from .conftest import make_hand

META = VideoMeta(path="x", name="x", width=1920, height=1080, fps=30.0, n_frames=10, duration_s=1 / 3, codec="h")


def _lm_result(hands: list[tuple[str, float, list[tuple[float, float]]]]) -> SimpleNamespace:
    """Fake MediaPipe result: [(side 'Left'/'Right', score, 21 normalised points)]."""
    return SimpleNamespace(
        hand_landmarks=[[SimpleNamespace(x=x, y=y) for x, y in pts] for _, _, pts in hands],
        handedness=[[SimpleNamespace(category_name=side, score=sc)] for side, sc, _ in hands])


def _backend(cfg: HandsConfig, main: SimpleNamespace, crop_results: list | None = None,
             boxes: list | None = None) -> tuple[MediaPipeHands, dict]:
    """MediaPipeHands with fake landmarkers and detector (no model files needed)."""
    b = MediaPipeHands.__new__(MediaPipeHands)
    b.cfg, b.meta, b.expected, b._last_ts = cfg, META, 2, -1
    b.counters = {"recovered": 0, "retries": 0, "dropped_other_person": 0,
                  "rtdetr_boxes": 0, "rtdetr_boxes_no_hand": 0, "rtdetr_side_disagree": 0}
    calls = {"detector": 0, "crops": []}
    b._lm = SimpleNamespace(detect_for_video=lambda img, ts: main)
    results = iter(crop_results or [])

    def crop_detect(img):
        calls["crops"].append(img.numpy_view().shape)
        return next(results)

    def detect(img):
        calls["detector"] += 1
        return list(boxes or [])

    b._retry_lm = SimpleNamespace(detect=crop_detect)
    b._detector = SimpleNamespace(detect=detect)
    return b, calls


FRAME = Frame(index=0, t=0.0, image=np.zeros((540, 960, 3), np.uint8), scale=0.5)
PTS = [(0.4 + 0.01 * i, 0.4 + 0.01 * i) for i in range(21)]  # a hand in the middle of the crop
RTDETR = HandsConfig(recover=True, recover_detector="rtdetr")


def test_square_crop_and_mapping_back_to_full_frame():
    crop = square_crop([100, 200, 140, 220], pad=0.5, w=960, h=540)
    assert crop == (90, 180, 150, 240)  # side = 40 * 1.5 = 60, centred on (120, 210)
    assert square_crop([0, 0, 40, 20], 0.5, 960, 540) == (0, 0, 60, 60)  # shifted inside, not shrunk
    assert square_crop([900, 500, 960, 540], 0.5, 960, 540) == (870, 450, 960, 540)
    assert square_crop([0, 0, 900, 900], 0.5, 960, 540) == (180, 0, 720, 540)  # shrunk only if larger, stays centred
    # normalised crop point -> processing pixels -> original pixels (scale 0.5 = 960/1920)
    assert crop_to_frame([(0.0, 0.0), (0.5, 1.0)], crop, 0.5) == [[180.0, 360.0], [240.0, 480.0]]


def test_accept_side_iou_and_score():
    left = make_hand("left", x=100, y=100)
    assert accept(make_hand("right", x=500, y=500, score=0.9), [left], 0.3, 0.6)
    assert not accept(make_hand("left", x=500, y=500, score=0.9), [left], 0.3, 0.6)  # side present
    assert not accept(make_hand("right", x=102, y=102, score=0.9), [left], 0.3, 0.6)  # overlaps
    assert not accept(make_hand("right", x=500, y=500, score=0.5), [left], 0.3, 0.6)  # low score


def test_detector_never_called_when_mediapipe_found_both_hands():
    main = _lm_result([("Left", 0.9, PTS), ("Right", 0.9, [(x + 0.3, y) for x, y in PTS])])
    b, calls = _backend(RTDETR, main)
    hands = b.process(FRAME)
    assert len(hands) == 2 and calls["detector"] == 0 and b.counters["retries"] == 0


def test_rtdetr_box_becomes_recovered_hand_with_full_frame_keypoints():
    main = _lm_result([("Left", 0.9, [(0.1 + 0.001 * i, 0.5) for i in range(21)])])  # only a left hand
    boxes = [("right", 0.8, [600.0, 300.0, 640.0, 340.0]),  # new hand
             ("right", 0.7, [800.0, 100.0, 840.0, 140.0])]  # box without a MediaPipe hand
    crop_results = [_lm_result([("Right", 0.95, PTS)]), _lm_result([])]
    b, calls = _backend(RTDETR, main, crop_results, boxes)
    hands = b.process(FRAME)
    assert [h.source for h in hands] == ["detected", "recovered"]
    rec = hands[1]
    crop = square_crop(boxes[0][2], RTDETR.rtdetr_crop_pad, 960, 540)
    assert rec.side == "right" and rec.score == 0.95
    assert rec.kp2d[0] == pytest.approx([(crop[0] + 0.4 * (crop[2] - crop[0])) / 0.5,
                                         (crop[1] + 0.4 * (crop[3] - crop[1])) / 0.5])
    assert calls["crops"] == [(60, 60, 3), (60, 60, 3)]
    assert b.counters == {"recovered": 1, "retries": 1, "dropped_other_person": 0, "rtdetr_boxes": 2,
                          "rtdetr_boxes_no_hand": 1, "rtdetr_side_disagree": 0}


def test_rtdetr_box_over_an_existing_hand_is_skipped_and_main_hands_untouched():
    main = _lm_result([("Left", 0.9, PTS)])
    b, calls = _backend(RTDETR, main, [], [("left", 0.9, [384.0, 216.0, 576.0, 324.0])])  # = the hand, at 0.5 scale
    first = b.process(FRAME)[0]
    b_hands = MediaPipeHands._to_hands(b, main, "detected")[0]
    assert first == b_hands and calls["crops"] == []  # overlapping box: no crop, main hand unchanged


# ---- real clip ---------------------------------------------------------------------------------------
CLIP = Path("data/raw/MTL-SKL00089-005.mp4")
BASELINE = Path("artifacts/regression/hands_default_SKL005_10s.json")  # captured before this change (git-ignored)
MP_WEIGHTS = Path("weights/hand_landmarker.task")


@pytest.mark.smoke
@pytest.mark.skipif(not (CLIP.is_file() and BASELINE.is_file() and MP_WEIGHTS.is_file()),
                    reason="needs the clip, the pre-change baseline and MediaPipe weights")
def test_default_config_output_identical_to_before_the_change():
    from egovision.hands import make_backend
    from egovision.ingest import FrameSource, probe

    cfg = Config.load(None, ["ingest.max_seconds=10", "ingest.cache_frames=false"])
    assert cfg.hands.recover is False and cfg.hands.recover_detector == "clahe"
    meta = probe(CLIP)
    out = []
    with make_backend(cfg.hands, meta, expected=2) as hb:
        for fr in FrameSource(meta, cfg.ingest):
            out.append({"frame": fr.index, "hands": [h.model_dump() for h in hb.process(fr)]})
    assert out == json.loads(BASELINE.read_text())["frames"]  # same hands, same order, same values


@pytest.mark.smoke
def test_rtdetr_detector_runs_on_a_real_frame():
    weights = Path(RTDETR.rtdetr_weights)
    if not (weights.is_file() and CLIP.is_file()):
        pytest.skip("needs RT-DETR weights (scripts/download_weights.py rtdetr_hand_detector_v1.onnx) and the clip")
    pytest.importorskip("onnxruntime")
    from egovision.hands.rtdetr import RTDetrHandDetector
    from egovision.ingest import decode_frames

    _, img = next(iter(decode_frames(str(CLIP), (960, 540), {30}, 30)))
    dets = RTDetrHandDetector(weights, 0.5).detect(img)
    assert all(side in ("left", "right") and 0.5 <= sc <= 1 and len(box) == 4 for side, sc, box in dets)
    assert dets, "the bench worker's hands are visible at 1 s"
