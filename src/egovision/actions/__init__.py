"""Task-step (action) stage: turns a clip (+ hand tracks) into a list of Actions."""

from __future__ import annotations

import logging
import os
from typing import Protocol

from egovision.config import ActionsConfig
from egovision.schema import Action, FrameHands, VideoMeta

log = logging.getLogger(__name__)


class ActionBackend(Protocol):
    name: str

    def segment(self, meta: VideoMeta, hands: list[FrameHands]) -> list[Action]: ...


def make_backend(cfg: ActionsConfig) -> ActionBackend:
    choice = cfg.backend
    if choice == "auto":
        choice = "gemini" if os.environ.get(cfg.api_key_env) else "offline"
        if choice == "offline":
            log.warning("actions: %s not set -> using the offline hand-presence baseline "
                        "(verbs will be 'other'); set the key to use Gemini", cfg.api_key_env)
    if choice == "gemini":
        from egovision.actions.gemini import GeminiActions

        return GeminiActions(cfg)
    from egovision.actions.offline import HandPresenceBaseline

    return HandPresenceBaseline(cfg)
