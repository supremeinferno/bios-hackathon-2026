"""Stages 7-8 (FORGE): check each action against hand evidence, then refine its boundaries.

Stage 7 statuses (nothing is dropped; uncertain items go to review):
  verified    claimed hand visible, motion fits the verb, attribution agrees
  corrected   evidence says a different hand did it: `hand` replaced, original kept in evidence
  unverified  claimed hand rarely visible, or motion contradicts the verb
  rejected    no hand visible at all during the action
Stage 8 boundary refinement: each start/end moves to the slowest-hand moment within +/- window
(a "motion valley": hands pause between steps). Original times are kept as `orig_start_s/orig_end_s`.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from egovision.config import VerifyConfig
from egovision.schema import Action, ContactSegment

SIDES = ("left", "right")


class Signals:
    """Column view of signals.csv rows for fast window queries."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.t = np.array([float(r["t"]) for r in rows]) if rows else np.zeros(0)
        self.vis = {s: np.array([float(r[f"{s}_visible"] or 0) for r in rows]) for s in SIDES}
        self.speed = {s: np.array([float(r[f"{s}_speed"]) if r[f"{s}_speed"] not in ("", None) else 0.0 for r in rows])
                      for s in SIDES}

    def window(self, a: float, b: float) -> np.ndarray:
        return (self.t >= a) & (self.t <= b)


def claimed(hand: str) -> tuple[str, ...]:
    return {"left": ("left",), "right": ("right",), "both": SIDES}.get(hand, ())


def _words(o: str) -> set[str]:
    return set(re.findall(r"[a-z]+", o.lower())) - {"a", "an", "the", "of"}


def hand_activity(sig: Signals, a: float, b: float) -> dict[str, float]:
    """FORGE fuse_hand evidence: mean speed while visible x share of time visible."""
    w = sig.window(a, b)
    out = {}
    for s in SIDES:
        v = sig.vis[s][w]
        out[s] = float((sig.speed[s][w] * v).sum() / max(v.sum(), 1)) * float(v.mean()) if v.size else 0.0
    return out


def verify_action(act: Action, sig: Signals, contacts: list[ContactSegment], cfg: VerifyConfig) -> Action:
    w = sig.window(act.start_s, act.end_s)
    ev: dict[str, Any] = {}
    flags: list[str] = []
    if not w.any():
        return act.model_copy(update={"status": "unverified", "flags": ["no_signal"], "evidence": ev})
    pres = {s: float(sig.vis[s][w].mean()) for s in SIDES}
    ev["presence"] = {s: round(v, 2) for s, v in pres.items()}
    if max(pres.values()) < cfg.reject_presence:
        return act.model_copy(update={"status": "rejected", "flags": ["no_hand_visible"], "evidence": ev})
    act_l = hand_activity(sig, act.start_s, act.end_s)
    ev["activity"] = {s: round(v, 3) for s, v in act_l.items()}
    # 1) attribution (FORGE fuse_hand): a one-hand claim whose hand is barely visible while the
    #    other hand is clearly active is corrected to the active hand
    new_hand, corrected = act.hand, False
    left, right = act_l["left"], act_l["right"]
    if act.hand in ("left", "right", "both") and max(left, right) > cfg.min_activity:
        ratio = max(left, right) / max(min(left, right), 1e-6)
        measured = "both" if ratio < cfg.both_ratio else ("left" if left > right else "right")
        ev["measured_hand"] = measured
        if act.hand in SIDES and measured in SIDES and measured != act.hand and pres[act.hand] < cfg.min_presence:
            new_hand, corrected = measured, True
            flags.append(f"hand_corrected_from_{act.hand}")
    hands = claimed(new_hand)
    # 2) presence of the (possibly corrected) hand
    status = "corrected" if corrected else "verified"
    if hands and min(pres[s] for s in hands) < cfg.min_presence:
        status = "unverified"
        flags.append("claimed_hand_rarely_visible")
    # 3) motion vs verb
    moving = max((act_l[s] for s in hands), default=max(act_l.values()))
    if act.verb in cfg.still_verbs and moving > cfg.still_max_speed:
        status = "unverified"
        flags.append("too_much_motion_for_verb")
    if act.verb in cfg.moving_verbs and moving < cfg.moving_min_speed:
        status = "unverified"
        flags.append("too_little_motion_for_verb")
    # object link: a contact on the same hand(s) overlapping the action with a matching object
    if act.object not in ("", "unknown"):
        link = [c for c in contacts if c.hand in (claimed(new_hand) or SIDES)
                and min(c.t_end, act.end_s) > max(c.t_start, act.start_s)]
        ev["object_contact"] = any(_words(c.object_name) & _words(act.object) for c in link)
    conf = act.confidence
    if status != "verified" and conf is not None:
        conf = round(conf * 0.5, 3)
    return act.model_copy(update={"status": status, "flags": flags, "evidence": ev, "hand": new_hand,
                                  "confidence": conf})


def snap_boundaries(act: Action, sig: Signals, cfg: VerifyConfig) -> Action:
    """Move start/end to the slowest-hand sample within +/- window (FORGE snap_boundary)."""
    if cfg.snap_window_s <= 0 or sig.t.size == 0:
        return act
    hands = claimed(act.hand) or SIDES

    def snap(t: float) -> float:
        w = sig.window(t - cfg.snap_window_s, t + cfg.snap_window_s)
        if w.sum() < 3:
            return t
        sp = np.max(np.stack([sig.speed[s][w] for s in hands]), axis=0)
        return float(sig.t[w][int(np.argmin(sp))])

    s, e = snap(act.start_s), snap(act.end_s)
    if e - s < cfg.min_len_after_snap_s:
        return act
    return act.model_copy(update={"start_s": round(s, 2), "end_s": round(e, 2),
                                  "orig_start_s": act.start_s, "orig_end_s": act.end_s})


def apply_profile(act: Action, profile: Any) -> Action:
    """Stage 2: a hand the worker profile does not list is not guessed; it becomes 'unknown'."""
    if profile is None or profile.allows(act.hand):
        return act
    return act.model_copy(update={"hand": "unknown", "flags": [*act.flags, f"hand_not_in_profile:{act.hand}"],
                                  "status": "corrected" if act.status in (None, "verified") else act.status})


def verify_and_refine(actions: list[Action], signal_rows: list[dict[str, Any]], contacts: list[ContactSegment],
                      cfg: VerifyConfig, profile: Any = None) -> list[Action]:
    sig = Signals(signal_rows)
    out = [apply_profile(verify_action(a, sig, contacts, cfg), profile) for a in actions]
    if cfg.snap:
        out = [snap_boundaries(a, sig, cfg) for a in out]
    return sorted(out, key=lambda a: (a.start_s, a.hand))


def status_counts(actions: list[Action]) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in actions:
        out[a.status or "none"] = out.get(a.status or "none", 0) + 1
    return out
