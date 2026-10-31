"""Offline M1 placeholder: segments where the worker's hands are visible.

This is NOT action recognition. Verb/object are always 'other'/'unknown'. It exists so the
pipeline runs end to end without an API key. The real offline backend lands in M4.
"""

from __future__ import annotations

from egovision.config import ActionsConfig
from egovision.schema import Action, FrameHands, VideoMeta


class HandPresenceBaseline:
    name = "offline-hand-presence"

    def __init__(self, cfg: ActionsConfig):
        self.cfg = cfg

    def segment(self, meta: VideoMeta, hands: list[FrameHands]) -> list[Action]:
        def label(fh: FrameHands) -> str:
            sides = {h.side for h in fh.hands}
            return "both" if len(sides) == 2 else (next(iter(sides)) if sides else "none")

        runs: list[list] = []  # [label, start_t, end_t]
        for fh in hands:
            lab = label(fh)
            end_t = fh.t + 1.0 / meta.fps
            if runs and runs[-1][0] == lab:
                runs[-1][2] = end_t
            else:
                runs.append([lab, fh.t, end_t])
        # drop very short runs by absorbing them into the previous run, then re-merge equal labels
        merged: list[list] = []
        for r in runs:
            if merged and (r[2] - r[1] < self.cfg.merge_gap_s or merged[-1][0] == r[0]):
                merged[-1][2] = r[2]
            else:
                merged.append(list(r))
        return [
            Action(clip=meta.name, start_s=round(s, 2), end_s=round(e, 2), verb="other", object="unknown",
                   hand=lab, confidence=None, source=self.name)
            for lab, s, e in merged
            if lab != "none" and e - s >= self.cfg.min_action_s
        ]
