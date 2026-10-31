"""Gemini video-understanding backend (PS3 minimum goal, PDF step 6), with the PDF's API rules:
key only from an env var, every response cached on disk, retries with backoff, uploads deleted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from egovision.actions.postprocess import from_raw
from egovision.config import ActionsConfig
from egovision.media import transcode
from egovision.schema import Action, FrameHands, VideoMeta

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1"


def build_prompt(verbs: list[str], duration_s: float) -> str:
    return f"""This is a {duration_s:.0f}-second first-person video from a camera mounted on a factory
worker's head. The worker's own hands are at the bottom of the view. The image is NOT mirrored:
the worker's left hand appears on the left side.

List every action the worker's hands perform, in time order, as a JSON array. Each item has:
- "start_s", "end_s": seconds from the start of the video (numbers, one decimal)
- "verb": exactly one of {json.dumps(verbs)}
- "object": the main thing handled, as a short lower-case noun phrase (e.g. "aluminium profile")
- "hand": "left", "right" or "both" (the worker's hands only; ignore other people)
- "confidence": your confidence from 0 to 1

Rules: ignore other people's hands; do not list periods where the hands do nothing; actions of
the same hand must not overlap; prefer short precise actions over one long vague one; use
"other" only when no verb fits. Output only the JSON array."""


class GeminiActions:
    name = "gemini"

    def __init__(self, cfg: ActionsConfig):
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{cfg.api_key_env} is not set; create a key in Google AI Studio and export it")
        from google import genai

        self.cfg = cfg
        self.client = genai.Client(api_key=key)

    def _cache_file(self, meta: VideoMeta, prompt: str) -> Path:
        st = os.stat(meta.path)
        h = hashlib.sha1(
            json.dumps([meta.name, st.st_size, int(st.st_mtime), meta.n_frames, self.cfg.gemini_model, self.cfg.video_fps,
                        self.cfg.upload_width, prompt, PROMPT_VERSION]).encode()
        ).hexdigest()[:16]
        return self.cfg.cache_dir / meta.name / f"gemini_{h}.json"

    def segment(self, meta: VideoMeta, hands: list[FrameHands]) -> list[Action]:
        prompt = build_prompt(self.cfg.verbs, meta.duration_s)
        cache = self._cache_file(meta, prompt)
        if cache.is_file():
            log.info("actions: using cached Gemini response %s", cache)
            raw = json.loads(cache.read_text(encoding="utf-8"))["parsed"]
        else:
            text = self._call(meta, prompt)
            raw = _parse_json_array(text)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"model": self.cfg.gemini_model, "prompt": prompt, "text": text,
                                         "parsed": raw, "created": time.time()}, indent=1), encoding="utf-8")
        return from_raw(raw, meta.name, self.cfg.verbs, source=f"gemini:{self.cfg.gemini_model}")

    def _call(self, meta: VideoMeta, prompt: str) -> str:
        from google.genai import types

        small = self.cfg.cache_dir / meta.name / f"upload_w{self.cfg.upload_width}_n{meta.n_frames}.mp4"
        if not small.is_file():
            log.info("actions: transcoding %s to width %d for upload", meta.name, self.cfg.upload_width)
            transcode(meta.path, small, self.cfg.upload_width, meta.fps, max_frames=meta.n_frames)
        uploaded = None
        try:
            uploaded = self._retry(lambda: self.client.files.upload(file=str(small)))
            while uploaded.state is not None and uploaded.state.name == "PROCESSING":
                time.sleep(5)
                uploaded = self.client.files.get(name=uploaded.name)
            if uploaded.state is not None and uploaded.state.name == "FAILED":
                raise RuntimeError(f"Gemini could not process the upload for {meta.name}")
            part = types.Part(
                file_data=types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
                video_metadata=types.VideoMetadata(fps=self.cfg.video_fps),
            )
            resp = self._retry(lambda: self.client.models.generate_content(
                model=self.cfg.gemini_model, contents=[part, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
            ))
            return resp.text or "[]"
        finally:
            if uploaded is not None and self.cfg.delete_uploads:
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception as e:  # noqa: BLE001 - deletion failure must not hide the result
                    log.warning("actions: could not delete uploaded file %s: %s", uploaded.name, e)

    def _retry(self, fn: Any) -> Any:
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 - SDK raises varied network/rate-limit errors
                if attempt == self.cfg.max_retries:
                    raise
                wait = 5 * 2**attempt
                log.warning("actions: Gemini call failed (%s); retry %d in %ds", e, attempt + 1, wait)
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
