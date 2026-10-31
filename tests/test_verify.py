from __future__ import annotations

from egovision.config import VerifyConfig
from egovision.schema import Action, ContactSegment
from egovision.verify import Signals, snap_boundaries, verify_action, verify_and_refine

CFG = VerifyConfig()


def rows(n=40, hz=10, left=(1, 0.0), right=(1, 0.5), pause_at=None):
    out = []
    for i in range(n):
        t = i / hz
        rs = right[1]
        if pause_at is not None and abs(t - pause_at) < 0.05:
            rs = 0.0
        out.append({"t": t, "left_visible": left[0], "left_speed": left[1] if left[0] else "",
                    "right_visible": right[0], "right_speed": rs if right[0] else ""})
    return out


def A(verb="wipe", hand="right", s=0.5, e=3.0, obj="granite slab"):  # noqa: N802
    return Action(clip="c", start_s=s, end_s=e, verb=verb, object=obj, hand=hand, confidence=0.8)


def test_verified_when_claimed_hand_visible_and_moving():
    a = verify_action(A(), Signals(rows()), [], CFG)
    assert a.status == "verified" and a.flags == [] and a.evidence["presence"]["right"] == 1.0


def test_rejected_when_no_hand_visible():
    a = verify_action(A(), Signals(rows(left=(0, 0), right=(0, 0))), [], CFG)
    assert a.status == "rejected" and a.flags == ["no_hand_visible"]


def test_unverified_when_motion_contradicts_verb():
    still = Signals(rows(right=(1, 0.0)))
    assert verify_action(A("wipe"), still, [], CFG).status == "unverified"
    fast = Signals(rows(right=(1, 2.0)))
    a = verify_action(A("hold"), fast, [], CFG)
    assert a.status == "unverified" and "too_much_motion_for_verb" in a.flags and a.confidence == 0.4


def test_hand_corrected_when_other_hand_did_it():
    sig = Signals(rows(left=(1, 0.5), right=(0, 0)))
    a = verify_action(A(hand="right"), sig, [], CFG)
    assert a.hand == "left" and a.status == "corrected" and "hand_corrected_from_right" in a.flags


def test_object_contact_evidence():
    c = ContactSegment(clip="c", t_start=0.0, t_end=2.0, hand="right", state="fixed", object_name="granite slab")
    assert verify_action(A(), Signals(rows()), [c], CFG).evidence["object_contact"] is True


def test_snap_to_motion_valley_keeps_original_times():
    sig = Signals(rows(n=60, pause_at=1.0))
    a = snap_boundaries(A(s=1.5, e=4.0), sig, CFG)
    assert a.start_s == 1.0 and a.orig_start_s == 1.5 and a.orig_end_s == 4.0
    out = verify_and_refine([A(s=1.5, e=4.0)], rows(n=60, pause_at=1.0), [], CFG)
    assert out[0].status == "verified" and out[0].start_s == 1.0


def test_csv_maps_none_to_unknown():
    assert A(hand="none").csv_row()[-1] == "unknown"


def test_profile_hand_not_listed_becomes_unknown(tmp_path):
    from egovision.profile import load_profile

    y = tmp_path / "p.yaml"
    y.write_text("sources:\n  one-handed:\n    manipulators:\n      - {id: left, kind: hand}\n"
                 "      - {id: right, kind: none}\n", encoding="utf-8")
    prof = load_profile(y, "one-handed-007")
    assert prof.expected == ["left"] and prof.allows("left") and not prof.allows("right")
    out = verify_and_refine([A(hand="right")], rows(), [], CFG, prof)
    assert out[0].hand == "unknown" and "hand_not_in_profile:right" in out[0].flags
    assert load_profile(y, "other-clip").expected == ["left", "right"]  # falls back to default
