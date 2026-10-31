from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from egovision.config import StepsConfig
from egovision.schema import Action
from egovision.steps.gemini import GeminiSteps, make_windows, pass2_targets, stitch


class FakeClient:
    """Uploads succeed; models in `fail_models` are 'overloaded'; others answer."""

    def __init__(self, fail_models: set[str]):
        self.fail_models, self.calls, self.deleted = fail_models, [], []
        self.files = SimpleNamespace(
            upload=lambda file: SimpleNamespace(name="files/1", uri="u", mime_type="video/mp4", state=None),
            get=lambda name: None, delete=lambda name: self.deleted.append(name))
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.calls.append(model)
        if model in self.fail_models:
            raise RuntimeError("503 UNAVAILABLE: high demand")
        return SimpleNamespace(text='[{"start_s": 0, "end_s": 1, "verb": "wipe", "object": "slab", "hand": "right"}]')


def _backend(tmp_path: Path, client) -> GeminiSteps:
    g = GeminiSteps.__new__(GeminiSteps)
    g.cfg = StepsConfig(cache_dir=tmp_path, max_retries=0, gemini_model="main", gemini_fallback_models=["fb1", "fb2"])
    g.client, g.model_used, g.models_used, g.exhausted = client, None, {}, set()
    g.source_video = tmp_path / "san.mp4"
    return g


def test_falls_back_to_next_model_and_deletes_upload(tmp_path: Path):
    c = FakeClient({"main"})
    text, model = _backend(tmp_path, c)._ask(tmp_path / "w.mp4", "prompt", 2.0)
    assert model == "fb1" and c.calls == ["main", "fb1"] and "wipe" in text and c.deleted == ["files/1"]


def test_all_models_failing_raises_and_still_deletes(tmp_path: Path):
    c = FakeClient({"main", "fb1", "fb2"})
    with pytest.raises(RuntimeError, match="all Gemini models failed"):
        _backend(tmp_path, c)._ask(tmp_path / "w.mp4", "prompt", 2.0)
    assert c.calls == ["main", "fb1", "fb2"] and c.deleted == ["files/1"]


def test_client_errors_are_not_retried(tmp_path: Path):
    class NotFound(Exception):
        code = 404

    c = FakeClient(set())
    calls = []

    def gen(model, contents, config):
        calls.append(model)
        if model == "main":
            raise NotFound("404 model no longer available")
        return SimpleNamespace(text="[]")

    c.models = SimpleNamespace(generate_content=gen)
    g = _backend(tmp_path, c)
    g.cfg = g.cfg.model_copy(update={"max_retries": 3})
    _, model = g._ask(tmp_path / "w.mp4", "prompt", 2.0)
    assert model == "fb1" and calls == ["main", "fb1"]  # 404 tried once, not 4 times


def test_windows_stitch_and_pass2_targets():
    assert make_windows(70, 30) == [(0.0, 30.0), (30.0, 60.0), (60.0, 70.0)]
    assert make_windows(10, 30) == [(0.0, 10.0)]

    def A(s, e, verb="wipe", hand="right", conf=0.9):  # noqa: N802
        return Action(clip="c", start_s=s, end_s=e, verb=verb, object="slab", hand=hand, confidence=conf)

    merged = stitch([A(20, 30), A(30.2, 40), A(30, 35, hand="left")], gap_s=1.0)
    assert [(a.hand, a.start_s, a.end_s) for a in merged] == [("right", 20, 40), ("left", 30, 35)]
    cfg = StepsConfig(pass2_window_s=10, pass2_max_windows=5)
    t = pass2_targets([A(10, 12, verb="other"), A(13, 14, conf=0.3), A(50, 52)], 60, cfg)
    assert t == [(6.0, 18.5)]  # two uncertain actions merged into one window; the confident one ignored


def test_quota_exhausted_model_is_skipped_for_the_rest_of_the_run(tmp_path: Path):
    class Quota(Exception):
        code = 429

    c = FakeClient(set())
    calls = []

    def gen(model, contents, config):
        calls.append(model)
        if model == "main":
            raise Quota("429 RESOURCE_EXHAUSTED: You exceeded your current quota")
        return SimpleNamespace(text="[]")

    c.models = SimpleNamespace(generate_content=gen)
    g = _backend(tmp_path, c)
    g.cfg = g.cfg.model_copy(update={"max_retries": 2})
    g._ask(tmp_path / "w.mp4", "p", 2.0)
    g._ask(tmp_path / "w.mp4", "p", 2.0)
    assert calls == ["main", "fb1", "fb1"]  # quota hit once, no retries, then skipped
