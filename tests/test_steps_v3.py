"""Audit fixes in the step stage: verb normalisation, idle items, boundary-only stitching, prompt v3,
structured output and model pinning."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from egovision.config import DEFAULT_VERB_GLOSSARY, DEFAULT_VERBS, StepsConfig
from egovision.schema import Action
from egovision.steps.gemini import GeminiSteps, build_prompt, response_schema, stitch
from egovision.steps.postprocess import from_raw, is_idle_item, normalise_verb


def A(s, e, verb="wipe", obj="slab", hand="right"):  # noqa: N802
    return Action(clip="c", start_s=s, end_s=e, verb=verb, object=obj, hand=hand)


def test_inflected_verbs_no_longer_fall_through_to_other():
    # before the audit these four returned "other" (startswith('place') is False for 'placing')
    for word, verb in [("placing", "place"), ("wiping", "wipe"), ("carrying", "carry"), ("writing", "write")]:
        assert normalise_verb(word, DEFAULT_VERBS) == verb


def test_synonyms_follow_spec_definitions_and_unknowns_stay_other():
    assert normalise_verb("wrap", DEFAULT_VERBS) == "attach"  # SPEC A2: tape/clips/end caps are attach
    assert normalise_verb("screwing", DEFAULT_VERBS) == "fasten"
    assert normalise_verb("put down", DEFAULT_VERBS) == "place"
    for word in ("pull", "reach", "roll", "walk"):
        assert normalise_verb(word, DEFAULT_VERBS) == "other"


def test_idle_items_detected():
    assert is_idle_item({"verb": "other", "object": "nothing"})
    assert not is_idle_item({"verb": "other", "object": "tape"})
    assert not is_idle_item({"verb": "hold", "object": "nothing"})


def test_from_raw_keeps_vlm_visibility_as_evidence():
    a = from_raw([{"start_s": 0, "end_s": 1, "verb": "press", "object": "remote", "hand": "right",
                   "hands_visible": "no"}], "c", DEFAULT_VERBS, "t")[0]
    assert a.evidence == {"vlm_hands_visible": "no"} and a.confidence is None


def test_boundary_stitch_rejoins_only_window_cuts():
    acts = [A(20, 29.8), A(30.1, 40, obj="slabs"),  # cut by the window edge at 30 s -> one action
            A(41, 45), A(45.5, 50)]  # two separate wipes inside one window -> stay two
    out = stitch(acts, gap_s=1.0, boundaries=[30.0, 60.0])
    assert [(a.start_s, a.end_s) for a in out] == [(20, 40), (41, 45), (45.5, 50)]
    # the original behaviour merges the in-window pair and misses the plural object
    old = stitch(acts, gap_s=1.0)
    assert [(a.start_s, a.end_s) for a in old] == [(20, 29.8), (30.1, 40), (41, 50)]


def test_v3_prompt_has_every_definition_and_schema_enums():
    p = build_prompt(DEFAULT_VERBS, 30, "v3-glossary", DEFAULT_VERB_GLOSSARY)
    for v in DEFAULT_VERBS:
        assert f"- {v}: " in p
    assert "hands_visible" in p and "confidence" not in p
    assert build_prompt(DEFAULT_VERBS, 30) == build_prompt(DEFAULT_VERBS, 30, "v2-window")  # v2 unchanged
    s = response_schema(DEFAULT_VERBS)
    assert s["items"]["properties"]["verb"]["enum"] == DEFAULT_VERBS
    assert s["items"]["properties"]["hand"]["enum"] == ["left", "right", "both"]


class _Client:
    def __init__(self, fail: set[str]):
        self.fail, self.calls, self.configs = fail, [], []
        self.files = SimpleNamespace(
            upload=lambda file: SimpleNamespace(name="f", uri="u", mime_type="video/mp4", state=None),
            get=lambda name: None, delete=lambda name: None)
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.calls.append(model)
        self.configs.append(config)
        if model in self.fail:
            raise RuntimeError("503 UNAVAILABLE")
        return SimpleNamespace(text="[]")


def _g(tmp_path: Path, client: _Client, **upd) -> GeminiSteps:
    g = GeminiSteps.__new__(GeminiSteps)
    g.cfg = StepsConfig(cache_dir=tmp_path, max_retries=0, gemini_model="main", gemini_fallback_models=["fb"], **upd)
    g.client, g.model_used, g.models_used, g.exhausted = client, None, {}, set()
    g.source_video = tmp_path / "san.mp4"
    return g


def test_pinned_model_never_falls_back(tmp_path: Path):
    c = _Client({"main"})
    try:
        _g(tmp_path, c, pin_model=True)._ask(tmp_path / "w.mp4", "p", 2.0)
        raise AssertionError("expected failure")
    except RuntimeError as e:
        assert "all Gemini models failed" in str(e)
    assert c.calls == ["main"]


def test_structured_output_and_seed_are_sent(tmp_path: Path):
    c = _Client(set())
    _g(tmp_path, c, structured_output=True, gemini_seed=7)._ask(tmp_path / "w.mp4", "p", 2.0)
    cfg = c.configs[0]
    assert cfg.response_json_schema["items"]["properties"]["verb"]["enum"] == DEFAULT_VERBS and cfg.seed == 7


def test_cache_key_unchanged_for_v2_and_model_specific_when_pinned(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    meta = SimpleNamespace(path=str(video), name="v")
    k_v2 = _g(tmp_path, _Client(set()))._cache_key(meta, 0, 30, 2.0, "p")  # type: ignore[arg-type]
    k_pin_a = _g(tmp_path, _Client(set()), pin_model=True)._cache_key(meta, 0, 30, 2.0, "p")  # type: ignore[arg-type]
    g_b = _g(tmp_path, _Client(set()), pin_model=True)
    g_b.cfg = g_b.cfg.model_copy(update={"gemini_model": "other"})
    k_pin_b = g_b._cache_key(meta, 0, 30, 2.0, "p")  # type: ignore[arg-type]
    assert len({k_v2, k_pin_a, k_pin_b}) == 3


def test_pinned_run_reuses_answer_cached_by_the_same_model_only(tmp_path: Path):
    import json

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    meta = SimpleNamespace(path=str(video), name="v")
    legacy = _g(tmp_path, _Client(set()))._cache_key(meta, 0, 30, 2.0, "p")  # type: ignore[arg-type]
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({"model": "main", "parsed": []}), encoding="utf-8")
    assert _g(tmp_path, _Client(set()), pin_model=True)._cached(meta, 0, 30, 2.0, "p") == legacy  # type: ignore[arg-type]
    other = _g(tmp_path, _Client(set()), pin_model=True)
    other.cfg = other.cfg.model_copy(update={"gemini_model": "fb"})
    assert other._cached(meta, 0, 30, 2.0, "p") is None  # type: ignore[arg-type]
