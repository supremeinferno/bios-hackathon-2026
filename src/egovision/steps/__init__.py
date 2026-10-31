"""Steps stage: per-hand task-step timeline from hand tracks, contacts and keyframe features."""

from __future__ import annotations

from typing import Protocol

from egovision.config import StepsConfig
from egovision.schema import Action, ContactSegment, FrameHands, VideoMeta
from egovision.steps.features import KeyframeFeatures


class StepBackend(Protocol):
    name: str
    needs_embeddings: bool

    def segment(self, meta: VideoMeta, hands: list[FrameHands], keyframes: list[KeyframeFeatures],
                contacts: list[ContactSegment]) -> list[Action]: ...


def make_backend(cfg: StepsConfig, device: str) -> StepBackend:
    if cfg.backend == "gemini":
        from egovision.steps.gemini import GeminiSteps

        return GeminiSteps(cfg)
    from egovision.steps.exemplar import ExemplarMatcher

    return ExemplarMatcher(cfg, device)
