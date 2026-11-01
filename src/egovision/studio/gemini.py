"""Gemini batch analyser: timestamped frames in, structured JSON of worker tasks out.

Same API rules as FORGE's Gemini step (egovision/steps/gemini.py): key only from GEMINI_API_KEY (environment or
.env), model fallbacks on overload or timeout, 4xx never retried, a model whose quota is exhausted is skipped for
the rest of the run. Frames go inline with the request, so nothing is stored in Gemini's file storage. The caller
saves every reply to disk.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from egovision.studio.analyzer import (
    SYSTEM_PROMPT,
    AnalyzerFatal,
    AnalyzerNotConfigured,
    BatchContext,
    BatchFailed,
    BatchResult,
    parts,
)
from egovision.studio.models import reply_schema
from egovision.studio.settings import StudioSettings

log = logging.getLogger(__name__)
BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}


def gemini_key() -> str | None:
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True), override=False)
    except ImportError:
        pass
    return os.environ.get("GEMINI_API_KEY") or None


def _code(e: Exception) -> int | None:
    c = getattr(e, "code", None)
    return c if isinstance(c, int) else None


class GeminiAnalyzer:
    def __init__(self, settings: StudioSettings) -> None:
        key = gemini_key()
        if not key:
            raise AnalyzerNotConfigured("GEMINI_API_KEY is not set in .env")
        from google import genai
        from google.genai import types

        self.s = settings
        self.model = settings.gemini_model
        self.models = list(dict.fromkeys([settings.gemini_model, *settings.gemini_fallback_models]))
        self.exhausted: set[str] = set()  # quota used up: skipped for the rest of this run
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(settings.api_timeout_s * 1000)))
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, response_mime_type="application/json", response_json_schema=reply_schema(),
            temperature=0.0, max_output_tokens=settings.max_output_tokens)

    def analyze(self, ctx: BatchContext) -> BatchResult:
        from google.genai import types

        contents: list[Any] = [v if kind == "text" else types.Part.from_bytes(data=v, mime_type="image/jpeg")
                               for kind, v in parts(ctx)]
        errors: list[str] = []
        for model in self.models:
            if model in self.exhausted:
                continue
            t0 = time.perf_counter()
            try:
                resp = self._call(model, contents)
            except AnalyzerFatal:
                raise
            except Exception as e:  # noqa: BLE001 - overload / timeout / quota: try the next model
                errors.append(f"{model}: {type(e).__name__} {_code(e) or ''} {str(e)[:160]}".strip())
                log.warning("gemini: %s failed for batch %d (%s); trying the next model", model, ctx.batch_index,
                            str(e)[:160])
                continue
            return self._result(model, resp, time.perf_counter() - t0)
        if all(m in self.exhausted for m in self.models):
            raise AnalyzerFatal("Gemini quota is exhausted on every model (free-tier limit): " + " | ".join(errors))
        raise BatchFailed("all Gemini models failed: " + " | ".join(errors))

    def _call(self, model: str, contents: list[Any]) -> Any:
        for attempt in range(self.s.max_retries + 1):
            try:
                return self.client.models.generate_content(model=model, contents=contents, config=self.config)
            except Exception as e:
                code, msg = _code(e), str(e).lower()
                if code in (401, 403) or (code == 400 and ("api key" in msg or "api_key" in msg)):
                    raise AnalyzerFatal(f"Gemini rejected the API key ({code}): {str(e)[:200]}") from e
                if code == 429 and "quota" in msg:
                    self.exhausted.add(model)
                    raise
                if (code is not None and 400 <= code < 500 and code != 429) or attempt == self.s.max_retries:
                    raise  # a 4xx never succeeds on retry; move to the next model
                wait = 5 * 2**attempt
                log.warning("gemini: %s busy (%s); retry %d in %d s", model, str(e)[:120], attempt + 1, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def _result(self, model: str, resp: Any, latency: float) -> BatchResult:
        raw = resp.model_dump(mode="json", exclude_none=True)
        raw.pop("sdk_http_response", None)
        cand = (resp.candidates or [None])[0]
        finish = getattr(getattr(cand, "finish_reason", None), "name", None)
        block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
        if block is not None or finish in BLOCKED:
            raise BatchFailed(f"Gemini blocked this batch ({getattr(block, 'name', block) or finish})", raw)
        if finish == "MAX_TOKENS":
            raise BatchFailed("reply was cut off at max_output_tokens; raise STUDIO_MAX_OUTPUT_TOKENS or lower "
                              "STUDIO_BATCH_FRAMES", raw)
        try:
            reply = json.loads(resp.text or "")
        except json.JSONDecodeError as e:
            raise BatchFailed(f"reply is not valid JSON: {e}", raw) from e
        if not isinstance(reply, dict):
            raise BatchFailed("reply is not a JSON object", raw)
        u = resp.usage_metadata
        n_out = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
        return BatchResult(reply=reply, raw=raw, model=model, input_tokens=getattr(u, "prompt_token_count", 0) or 0,
                           output_tokens=n_out, latency_s=round(latency, 2))
