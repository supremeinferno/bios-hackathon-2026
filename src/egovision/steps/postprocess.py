"""Backend-independent clean-up so every backend emits the same, valid timeline."""

from __future__ import annotations

import re
from typing import Any

from egovision.schema import Action

_HANDS = {"left": "left", "right": "right", "both": "both", "none": "none",
          "l": "left", "r": "right", "both hands": "both", "": "none", "neither": "none"}


def parse_time(v: Any) -> float:
    """Seconds from a number, '12.5', 'MM:SS(.f)' or 'HH:MM:SS(.f)'."""
    if isinstance(v, int | float):
        return float(v)
    s = str(v).strip().lower().rstrip("s")
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    return float(s)


def normalise_verb(v: str, verbs: list[str]) -> str:
    v = (v or "").strip().lower()
    if v in verbs:
        return v
    for cand in verbs:  # "picking up" -> "pick", "fastens" -> "fasten"
        if v.startswith(cand) or cand in v.split():
            return cand
    return "other" if "other" in verbs else verbs[-1]


def normalise_object(o: str) -> str:
    o = re.sub(r"\s+", " ", (o or "").strip().lower())
    o = re.sub(r"^(the|a|an) ", "", o)
    return o or "unknown"


_ADJ = {"wooden": "wood", "metallic": "metal", "plastic": "plastic"}
_GENERIC = {"piece", "pieces", "part", "parts", "small", "large", "big", "some", "bit", "item"}


def canonical_object(o: str) -> str:
    """Grouping key for objects, e.g. 'wooden piece' == 'wood piece' == 'piece of wood' -> 'wood'.

    Used to group steps (SOP templates, cycles); exported object names are left as the model wrote them.
    """
    t = normalise_object(o)
    t = re.split(r" (?:to|with|on|into|onto|from|for|using|in) ", t)[0]
    m = re.match(r"^(?:piece|pieces|bit|part) of (.+)$", t)
    if m:
        t = m.group(1)
    words = []
    for w in t.split():
        w = _ADJ.get(w, w)
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    kept = [w for w in words if w not in _GENERIC] or words
    return " ".join(kept) or "unknown"


def clean(actions: list[Action], duration_s: float, min_len_s: float = 0.2) -> list[Action]:
    out = []
    for a in actions:
        s, e = max(0.0, a.start_s), min(duration_s, a.end_s)
        if e - s < min_len_s:
            continue
        out.append(a.model_copy(update={"start_s": round(s, 2), "end_s": round(e, 2)}))
    return sorted(out, key=lambda a: (a.start_s, a.end_s))


def from_raw(raw: list[dict[str, Any]], clip: str, verbs: list[str], source: str) -> list[Action]:
    """Build Actions from loosely-typed model output; skip items that cannot be repaired."""
    out = []
    for item in raw:
        try:
            s, e = parse_time(item.get("start_s", item.get("start"))), parse_time(item.get("end_s", item.get("end")))
        except (TypeError, ValueError):
            continue
        if e <= s:
            continue
        hand = _HANDS.get(str(item.get("hand", "")).strip().lower(), "none")
        conf = item.get("confidence")
        out.append(Action(clip=clip, start_s=s, end_s=e, verb=normalise_verb(str(item.get("verb", "")), verbs),
                          object=normalise_object(str(item.get("object", ""))), hand=hand,  # type: ignore[arg-type]
                          confidence=float(conf) if isinstance(conf, int | float) and 0 <= conf <= 1 else None,
                          source=source))
    return out
