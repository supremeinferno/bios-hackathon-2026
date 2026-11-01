"""Objects stage: open-vocabulary detection on keyframes + hand-object contact (PS2 rule, PDF step 5)."""

from __future__ import annotations

from egovision.objects.contact import associate, contact_segments, nearest_object
from egovision.objects.detector import OpenVocabDetector

__all__ = ["OpenVocabDetector", "associate", "contact_segments", "nearest_object"]
