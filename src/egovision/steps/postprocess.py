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


# Out-of-list verbs a model or annotator may write, mapped by the SPEC A2 definitions (config
# DEFAULT_VERB_GLOSSARY). Verbs with no fitting class (pull, open, fold, reach, walk) stay 'other'.
VERB_SYNONYMS: dict[str, str] = {
    "grab": "pick", "take": "pick", "lift": "pick",
    "put": "place", "set": "place", "drop": "place", "lay": "place",
    "grip": "hold", "steady": "hold", "support": "hold",
    "transport": "carry",
    "tape": "attach", "wrap": "attach", "clip": "attach", "glue": "attach", "stick": "attach", "fit": "attach",
    "screw": "fasten", "nail": "fasten", "drill": "fasten", "bolt": "fasten", "staple": "fasten", "hammer": "fasten",
    "saw": "cut", "trim": "cut", "plane": "sand", "polish": "grind", "file": "grind",
    "clean": "wipe", "scrub": "wipe", "rub": "wipe",
    "slide": "push", "tilt": "push",  # not "roll": "roll paper around parts" (SKL) is attach, a slab is push
    "switch": "press", "operate": "press",
    "align": "adjust", "position": "adjust", "tighten": "adjust",
    "check": "inspect", "examine": "inspect", "mark": "write",
}


def _forms(w: str) -> list[str]:
    """Candidate base forms: placing -> place, wiping -> wipe, carrying -> carry, cutting -> cut, taped -> tape."""
    out = [w]
    if w.endswith("ing") and len(w) > 5:
        b = w[:-3]
        out += [b, b + "e"] + ([b[:-1]] if len(b) > 2 and b[-1] == b[-2] else [])
    if w.endswith("ies"):
        out.append(w[:-3] + "y")
    if w.endswith("ed") and len(w) > 4:
        out += [w[:-2], w[:-1]] + ([w[:-3]] if w[-3] == w[-4] else [])
    if w.endswith("es"):
        out.append(w[:-2])
    if w.endswith("s") and not w.endswith("ss"):
        out.append(w[:-1])
    return out


def normalise_verb(v: str, verbs: list[str]) -> str:
    """Map free text to the closed verb list: exact, then inflected forms ('placing' -> place; before this
    audit 'placing', 'wiping', 'carrying' and 'writing' fell through to 'other'), then synonyms."""
    v = re.sub(r"\s+", " ", (v or "").strip().lower())
    if v in verbs:
        return v
    for w in v.split():
        forms = _forms(w)
        for f in forms:
            if f in verbs:
                return f
        for f in forms:
            if f in VERB_SYNONYMS and VERB_SYNONYMS[f] in verbs:
                return VERB_SYNONYMS[f]
    return "other" if "other" in verbs else verbs[-1]


_IDLE_OBJECTS = {"nothing", "none", "no object", "empty", "air", "idle", "n/a", ""}


def is_idle_item(item: dict[str, Any]) -> bool:
    """A model item that describes idle time ('other' on 'nothing'), which the prompt asks it not to list."""
    verb = str(item.get("verb", "")).strip().lower()
    obj = str(item.get("object", "")).strip().lower()
    return verb in ("other", "idle", "none", "rest") and obj in _IDLE_OBJECTS


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
        vis = str(item.get("hands_visible", "")).strip().lower()
        out.append(Action(clip=clip, start_s=s, end_s=e, verb=normalise_verb(str(item.get("verb", "")), verbs),
                          object=normalise_object(str(item.get("object", ""))), hand=hand,  # type: ignore[arg-type]
                          confidence=float(conf) if isinstance(conf, int | float) and 0 <= conf <= 1 else None,
                          source=source,
                          evidence={"vlm_hands_visible": vis} if vis in ("yes", "partly", "no") else None))
    return out
