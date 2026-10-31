"""GPU if available, CPU otherwise."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def resolve_device(pref: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        log.warning("device: torch not installed; using CPU")
        return "cpu"
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        log.warning("device: CUDA requested but not available; falling back to CPU")
    return "cpu"
