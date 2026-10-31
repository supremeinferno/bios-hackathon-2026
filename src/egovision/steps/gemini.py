"""Gemini video-understanding backend (PS3 minimum goal, PDF step 6; FORGE Stage 5), with the PDF's
API rules: key only from an env var, every response cached on disk, uploads deleted after use, and
only the SANITIZED copy ever uploaded.

Windowed, two passes (docs/DECISIONS.md D14). Whole 3-minute clips returned 503 "high demand" on
every model for an hour, while 10-second windows answered in ~4 s, so:
  pass 1  fixed windows (`window_s`) at `video_fps`; times shifted back to clip time; actions that
          continue across a window boundary are stitched
  pass 2  windows around uncertain actions (verb "other" or low confidence) re-asked at `pass2_fps`;
          their actions replace pass-1 actions inside the window
Model order: main model, then fallbacks on overload/timeout; 4xx errors are not retried.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from egovision.config import StepsConfig
from egovision.ingest import iter_all_frames
from egovision.media import write_video
from egovision.schema import Action, ContactSegment, FrameHands, VideoMeta
from egovision.steps.features import KeyframeFeatures
from egovision.steps.postprocess import canonical_object, from_raw, is_idle_item

log = logging.getLogger(__name__)

PROMPT_VERSION = "v2-window"


def build_prompt(verbs: list[str], duration_s: float, version: str = PROMPT_VERSION,
                 glossary: dict[str, str] | None = None) -> str:
    if version == "v3-glossary":
        return build_prompt_v3(verbs, duration_s, glossary or {})
    return f"""This is a {duration_s:.0f}-second first-person video from a camera mounted on a factory
worker's head. The worker's own hands are at the bottom of the view. The image is NOT mirrored:
the worker's left hand appears on the left side. Some regions are pixelated for privacy; ignore them.

List every action the worker's hands perform, in time order, as a JSON array. Each item has:
- "start_s", "end_s": seconds from the start of THIS video (numbers, one decimal)
- "verb": exactly one of {json.dumps(verbs)}
- "object": the main thing handled, as a short lower-case noun phrase (e.g. "aluminium profile")
- "hand": "left", "right" or "both" (the worker's hands only; ignore other people)
- "confidence": your confidence from 0 to 1

Rules: ignore other people's hands; do not list periods where the hands do nothing; do not invent
actions you cannot see; actions of the same hand must not overlap; prefer short precise actions over
one long vague one; use "other" only when no verb fits. Output only the JSON array."""


def build_prompt_v3(verbs: list[str], duration_s: float, glossary: dict[str, str]) -> str:
    """v3: verb definitions from SPEC A2, 'both' only for a shared action, hands_visible instead of confidence."""
    defs = "\n".join(f"- {v}: {glossary.get(v, v)}" for v in verbs)
    return f"""This is a {duration_s:.0f}-second first-person video from a fisheye camera on a factory worker's head.
The worker's own hands and forearms enter from the bottom of the view and may be only partly visible at the
frame edge. The image is NOT mirrored: the worker's left hand appears on the left side. Some regions are
pixelated for privacy; ignore them. Ignore other people's hands.

List the actions the worker's own hands perform, in time order. Use exactly one verb per action; the
definitions decide between close verbs:
{defs}

For each action give:
- "start_s", "end_s": seconds from the start of THIS video, one decimal
- "verb": one verb from the list
- "object": the main thing handled, a short lower-case noun phrase; use the same name for the same object
- "hand": "left", "right", or "both" only when both hands perform this same action together
- "hands_visible": "yes" if the acting hand is clearly in view, "partly" if only part of it or only part of
  the time, "no" if you infer the action without seeing the hand

Rules:
- Do not list time when the hands do nothing, rest, or move without handling anything.
- Do not invent actions you cannot see.
- Actions of the same hand must not overlap. One continuous activity (e.g. wiping for 20 s) is one action.
- Different hands may do different actions at the same time (e.g. left holds while right wipes): list both.
Output only the JSON array."""


def response_schema(verbs: list[str]) -> dict[str, Any]:
    return {"type": "array", "items": {
        "type": "object",
        "properties": {
            "start_s": {"type": "number"}, "end_s": {"type": "number"},
            "verb": {"type": "string", "enum": list(verbs)}, "object": {"type": "string"},
            "hand": {"type": "string", "enum": ["left", "right", "both"]},
            "hands_visible": {"type": "string", "enum": ["yes", "partly", "no"]},
        },
        "required": ["start_s", "end_s", "verb", "object", "hand", "hands_visible"],
    }}


def make_windows(duration_s: float, window_s: float) -> list[tuple[float, float]]:
    out, t = [], 0.0
    while t < duration_s - 0.5:
        out.append((round(t, 2), round(min(duration_s, t + window_s), 2)))
        t += window_s
    return out or [(0.0, duration_s)]


def stitch(actions: list[Action], gap_s: float, boundaries: list[float] | None = None) -> list[Action]:
    """Merge consecutive same-hand actions with the same verb and object that touch.

    boundaries=None (original behaviour): any such pair merges, including two separate actions inside one
    window. With the window edges given, only a pair that meets at an edge merges (both ends within gap_s of
    it), objects compared in canonical form ('wooden piece' == 'wood piece'), so an action cut by the window
    plan is rejoined and two real repetitions stay two actions.
    """
    out: list[Action] = []
    for a in sorted(actions, key=lambda x: (x.hand, x.start_s)):
        p = out[-1] if out else None
        if p and p.hand == a.hand and p.verb == a.verb and a.start_s - p.end_s <= gap_s:
            if boundaries is None:
                ok = p.object == a.object
            else:
                ok = canonical_object(p.object) == canonical_object(a.object) and any(
                    abs(p.end_s - b) <= gap_s and abs(a.start_s - b) <= gap_s for b in boundaries)
            if ok:
                out[-1] = p.model_copy(update={"end_s": max(p.end_s, a.end_s)})
                continue
        out.append(a)
    return sorted(out, key=lambda x: (x.start_s, x.hand))


def pass2_targets(actions: list[Action], duration_s: float, cfg: StepsConfig) -> list[tuple[float, float]]:
    """Windows around uncertain actions, merged, longest first, capped at pass2_max_windows."""
    spans = []
    for a in actions:
        low = a.confidence is not None and a.confidence < cfg.pass2_min_confidence
        if a.verb == "other" or low:
            c = (a.start_s + a.end_s) / 2
            half = max(cfg.pass2_window_s, a.end_s - a.start_s + 2) / 2
            spans.append([max(0.0, c - half), min(duration_s, c + half)])
    spans.sort()
    merged: list[list[float]] = []
    for s in spans:
        if merged and s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(s)
    return [(round(s, 2), round(e, 2)) for s, e in merged][: cfg.pass2_max_windows]


class GeminiSteps:
    name = "gemini"
    needs_embeddings = False

    def __init__(self, cfg: StepsConfig):
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{cfg.api_key_env} is not set; create a key in Google AI Studio and export it")
        from google import genai
        from google.genai import types

        self.cfg = cfg
        self.client = genai.Client(api_key=key,
                                   http_options=types.HttpOptions(timeout=int(cfg.request_timeout_s * 1000)))
        self.model_used: str | None = None
        self.models_used: dict[str, int] = {}
        self.exhausted: set[str] = set()  # models that reported 429 quota exhaustion: skipped for this run
        self.source_video: Path | None = None  # set by the pipeline to the sanitized copy

    # ---- orchestration ----------------------------------------------------------------------
    def segment(self, meta: VideoMeta, hands: list[FrameHands], keyframes: list[KeyframeFeatures],
                contacts: list[ContactSegment]) -> list[Action]:
        src = self._source(meta)
        p1 = make_windows(meta.duration_s, self.cfg.window_s)
        edges = [e for _, e in p1[:-1]] if self.cfg.stitch_mode == "boundary" else None
        acts = stitch(self._windows(meta, src, p1, self.cfg.video_fps, 1), self.cfg.stitch_gap_s, edges)
        if self.cfg.pass2:
            targets = pass2_targets(acts, meta.duration_s, self.cfg)
            if targets:
                log.info("steps: pass 2 on %d window(s): %s", len(targets), targets)
                redo = self._windows(meta, src, targets, self.cfg.pass2_fps, 2)
                inside = [a for a in acts if any(s <= a.start_s and a.end_s <= e for s, e in targets)]
                acts = stitch([a for a in acts if a not in inside] + redo, self.cfg.stitch_gap_s)
        self.model_used = max(self.models_used, key=self.models_used.__getitem__) if self.models_used else None
        return acts

    def _source(self, meta: VideoMeta) -> Path:
        if self.source_video is not None:
            return self.source_video
        if not self.cfg.allow_unsanitized_upload:
            raise RuntimeError("privacy: no sanitized copy of this clip, refusing to upload the original "
                               "(enable privacy, or set steps.allow_unsanitized_upload=true deliberately)")
        return Path(meta.path)

    def _cache_key(self, meta: VideoMeta, s: float, e: float, fps: float, prompt: str) -> Path:
        st = os.stat(meta.path)
        parts: list[Any] = [meta.name, st.st_size, int(st.st_mtime), s, e, fps, self.cfg.upload_width, prompt,
                            self.cfg.prompt_version]
        # New options extend the key only when used, so v2 answers cached before they existed still hit.
        if self.cfg.structured_output or self.cfg.gemini_seed is not None:
            parts += [self.cfg.structured_output, self.cfg.gemini_seed]
        if self.cfg.pin_model:
            parts.append(self.cfg.gemini_model)
        h = hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:16]
        return self.cfg.cache_dir / meta.name / f"gemini_{s:07.2f}_{e:07.2f}_{h}.json"

    def _cached(self, meta: VideoMeta, s: float, e: float, fps: float, prompt: str) -> Path | None:
        """Cached answer for this request. A pinned run also accepts an answer cached before pinning existed
        if that answer came from the pinned model (same prompt, same window, temperature 0)."""
        key = self._cache_key(meta, s, e, fps, prompt)
        if key.is_file():
            return key
        if self.cfg.pin_model:
            legacy = self.cfg.model_copy(update={"pin_model": False})
            old = GeminiSteps._cache_key(SimpleNamespace(cfg=legacy), meta, s, e, fps, prompt)  # type: ignore[arg-type]
            if old.is_file() and json.loads(old.read_text(encoding="utf-8")).get("model") == self.cfg.gemini_model:
                return old
        return None

    def _windows(self, meta: VideoMeta, src: Path, wins: list[tuple[float, float]], fps: float,
                 pass_no: int) -> list[Action]:
        prompts = {w: build_prompt(self.cfg.verbs, w[1] - w[0], self.cfg.prompt_version, self.cfg.verb_glossary)
                   for w in wins}
        todo = [w for w in wins if self._cached(meta, *w, fps, prompts[w]) is None]
        clips = self._cut(meta, src, todo) if todo else {}
        out: list[Action] = []
        for w in wins:
            s, e = w
            cache = self._cache_key(meta, s, e, fps, prompts[w])
            hit = self._cached(meta, s, e, fps, prompts[w])
            if hit is not None:
                cached = json.loads(hit.read_text(encoding="utf-8"))
                raw, model = cached["parsed"], cached["model"]
            else:
                text, model = self._ask(clips[w], prompts[w], fps)
                raw = _parse_json_array(text)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"model": model, "window": [s, e], "fps": fps, "pass": pass_no,
                                             "prompt": prompts[w], "text": text, "parsed": raw,
                                             "created": time.time()}, indent=1), encoding="utf-8")
                clips[w].unlink(missing_ok=True)
            self.models_used[model] = self.models_used.get(model, 0) + 1
            if self.cfg.drop_idle_items:
                raw = [r for r in raw if not is_idle_item(r)]
            for a in from_raw(raw, meta.name, self.cfg.verbs, source=f"gemini:{model}:p{pass_no}"):
                a_s, a_e = min(e, s + a.start_s), min(e, s + a.end_s)
                if a_e - a_s >= 0.2:
                    out.append(a.model_copy(update={"start_s": round(a_s, 2), "end_s": round(a_e, 2)}))
        log.info("steps: pass %d: %d window(s) -> %d actions", pass_no, len(wins), len(out))
        return out

    def _cut(self, meta: VideoMeta, src: Path, wins: list[tuple[float, float]]) -> dict[tuple[float, float], Path]:
        """Cut all windows from the (sanitized) source in one decode pass."""
        tag = "san" if self.source_video else "orig"
        d = self.cfg.cache_dir / meta.name
        d.mkdir(parents=True, exist_ok=True)
        paths = {w: d / f"win_{w[0]:07.2f}_{w[1]:07.2f}_{tag}.mp4" for w in wins}
        frames: dict[tuple[float, float], list] = {w: [] for w in wins}
        last = int(max(e for _, e in wins) * meta.fps)
        for i, img in iter_all_frames(str(src), self.cfg.upload_width, last):
            t = i / meta.fps
            for w in wins:
                if w[0] <= t < w[1]:
                    frames[w].append(img)
        for w in wins:
            write_video(frames[w], paths[w], meta.fps, bitrate=1_200_000)
        return paths

    # ---- one request with model fallback --------------------------------------------------------
    def _ask(self, upload: Path, prompt: str, fps: float) -> tuple[str, str]:
        from google.genai import types

        uploaded = None
        try:
            uploaded = self._retry(lambda: self.client.files.upload(file=str(upload)))
            while uploaded.state is not None and uploaded.state.name == "PROCESSING":
                time.sleep(2)
                uploaded = self.client.files.get(name=uploaded.name)
            if uploaded.state is not None and uploaded.state.name == "FAILED":
                raise RuntimeError(f"Gemini could not process the upload {upload.name}")
            part = types.Part(file_data=types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
                              video_metadata=types.VideoMetadata(fps=fps))
            errors: list[str] = []
            gen: dict[str, Any] = {"response_mime_type": "application/json", "temperature": 0.0}
            if self.cfg.structured_output:
                gen["response_json_schema"] = response_schema(self.cfg.verbs)
            if self.cfg.gemini_seed is not None:
                gen["seed"] = self.cfg.gemini_seed
            models = [self.cfg.gemini_model]
            if not self.cfg.pin_model:
                models += self.cfg.gemini_fallback_models
            for model in models:
                if model in self.exhausted:
                    continue
                try:
                    resp = self._retry(lambda m=model: self.client.models.generate_content(
                        model=m, contents=[part, prompt], config=types.GenerateContentConfig(**gen)))
                    if model != self.cfg.gemini_model:
                        log.warning("steps: answered by fallback model %s", model)
                    return resp.text or "[]", model
                except Exception as e:  # noqa: BLE001 - overload/timeout: try the next model
                    if getattr(e, "code", None) == 429 and "quota" in str(e).lower():
                        self.exhausted.add(model)  # daily/free-tier quota: retrying later in this run is futile
                    errors.append(f"{model}: {type(e).__name__}: {str(e)[:120]}")
                    log.warning("steps: model %s failed (%s); trying next model", model, str(e)[:160])
            raise RuntimeError("all Gemini models failed: " + " | ".join(errors))
        finally:
            if uploaded is not None and self.cfg.delete_uploads:
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception as e:  # noqa: BLE001 - deletion failure must not hide the result
                    log.warning("steps: could not delete uploaded file %s: %s", uploaded.name, e)

    def _retry(self, fn: Any) -> Any:
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 - SDK raises varied network/rate-limit errors
                code = getattr(e, "code", None)
                quota = code == 429 and "quota" in str(e).lower()
                if attempt == self.cfg.max_retries or quota or (isinstance(code, int) and 400 <= code < 500 and code != 429):
                    raise  # 4xx (except rate limit) never succeeds on retry: move on
                wait = 5 * 2**attempt
                log.warning("steps: Gemini call failed (%s); retry %d in %ds", str(e)[:160], attempt + 1, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1:] if "\n" in t else t
    start, end = t.find("["), t.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"no JSON array in model output: {text[:200]!r}")
    data = json.loads(t[start:end + 1])
    return [d for d in data if isinstance(d, dict)]
