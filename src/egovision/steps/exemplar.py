"""Offline step recogniser: match each hand at each keyframe to its nearest verb prototype.

Prototypes per verb (no labels needed for the first two):
  * text      - SigLIP text embeddings of "a first-person photo of hands <verb phrase> <object>"
  * cue       - prior over verbs given the held/touched object (config `object_verbs`)
  * visual    - mean SigLIP crop embedding of labelled exemplar segments (few-shot, optional)
Scores are combined in log space, then decoded per hand with Viterbi (switch penalty), short
segments are dropped, pick/place are carved at grab/release and L+R duplicates merge to 'both'.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from egovision.config import StepsConfig
from egovision.schema import Action, ContactSegment, FrameHands, VideoMeta
from egovision.steps.features import SIDES, KeyframeFeatures, SiglipEmbedder
from egovision.steps.temporal import carve_pick_place, drop_short, log_softmax, merge_both, runs, viterbi

log = logging.getLogger(__name__)
GENERIC_OBJECT = "an object"


class ExemplarMatcher:
    name = "exemplar"
    needs_embeddings = True

    def __init__(self, cfg: StepsConfig, device: str):
        self.cfg = cfg
        self.embedder = SiglipEmbedder(cfg.embed_model, device)
        self._text: dict[str, np.ndarray] = {}
        self.visual: np.ndarray | None = None  # (V, D) or None
        if cfg.exemplars_file is not None:
            self.visual = self._load_visual_prototypes(cfg.exemplars_file)

    # ---- prototypes -------------------------------------------------------------------------
    def text_prototypes(self, obj: str | None) -> np.ndarray:
        key = obj or GENERIC_OBJECT
        if key not in self._text:
            protos = []
            for v in self.cfg.verbs:
                phrases = self.cfg.verb_phrases.get(v, [f"{v} {{o}}"])
                e = self.embedder.texts([f"a first-person photo of hands {p.format(o=key)}" for p in phrases])
                m = e.mean(axis=0)
                protos.append(m / np.linalg.norm(m))
            self._text[key] = np.stack(protos)
        return self._text[key]

    def _load_visual_prototypes(self, path: Path) -> np.ndarray | None:
        """Few-shot: rows of clip,start_s,end_s,verb,object,hand; features from outputs/<clip>/features.npz."""
        sums: dict[str, list[np.ndarray]] = {v: [] for v in self.cfg.verbs}
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            feat = Path("outputs") / r["clip"] / "features.npz"
            if not feat.is_file() or r["verb"] not in sums:
                continue
            z = np.load(feat)
            t = z["t"]
            sel = (t >= float(r["start_s"])) & (t <= float(r["end_s"]))
            for side in SIDES if r["hand"] in ("both", "none") else (r["hand"],):
                e = z[side][sel]
                sums[r["verb"]] += [x for x in e if not np.isnan(x).any()]
        have = {v: np.mean(e, axis=0) for v, e in sums.items() if e}
        if not have:
            log.warning("steps: exemplars file %s gave no usable features; visual prototypes off", path)
            return None
        dim = next(iter(have.values())).shape[0]
        protos = np.stack([have.get(v, np.zeros(dim)) for v in self.cfg.verbs])
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        log.info("steps: visual prototypes for %d/%d verbs from %s", len(have), len(self.cfg.verbs), path)
        return np.where(norms > 0, protos / np.maximum(norms, 1e-9), 0.0)

    # ---- scoring ----------------------------------------------------------------------------
    def verb_logprobs(self, emb: np.ndarray | None, obj: str | None) -> np.ndarray:
        c = self.cfg
        lp = np.zeros(len(c.verbs))
        if emb is not None and c.w_text > 0:
            lp += c.w_text * log_softmax(self.text_prototypes(obj) @ emb / c.text_temperature)
        if obj and c.w_cue > 0:
            prior = c.object_verbs.get(obj, {})
            p = np.array([prior.get(v, 0.0) for v in c.verbs]) + 0.05
            lp += c.w_cue * np.log(p / p.sum())
        if emb is not None and self.visual is not None and c.w_visual > 0:
            sims = self.visual @ emb
            sims[np.linalg.norm(self.visual, axis=1) == 0] = -1.0
            lp += c.w_visual * log_softmax(sims / c.text_temperature)
        return log_softmax(lp)

    # ---- decoding -----------------------------------------------------------------------------
    def segment(self, meta: VideoMeta, hands: list[FrameHands], keyframes: list[KeyframeFeatures],
                contacts: list[ContactSegment]) -> list[Action]:
        c = self.cfg
        if not keyframes:
            return []
        times = np.array([kf.t for kf in keyframes])
        step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        idle = len(c.verbs)
        actions: list[Action] = []
        for side in SIDES:
            em = np.zeros((len(keyframes), idle + 1))
            vlp = np.full((len(keyframes), idle), np.nan)
            objs: list[str | None] = []
            for k, kf in enumerate(keyframes):
                sf = kf.sides.get(side)
                obj = sf.contact.object if sf and sf.contact and sf.contact.state != "none" else None
                objs.append(obj)
                if sf is None or not sf.present:
                    em[k, :idle], em[k, idle] = -c.absent_cost, 0.0
                    continue
                lp = self.verb_logprobs(sf.emb, obj)
                vlp[k] = lp
                if obj is not None:
                    p_idle = c.idle_prob_contact
                else:
                    p_idle = c.idle_prob_empty_slow if sf.speed < c.idle_speed else c.idle_prob_empty_moving
                em[k, :idle] = lp + np.log(1 - p_idle)
                em[k, idle] = np.log(p_idle)
            path = viterbi(em, c.switch_penalty)
            for lab, a, b in drop_short(runs(path), times, step, c.min_segment_s, idle):
                if lab == idle:
                    continue
                run_objs = [o for o in objs[a:b + 1] if o]
                obj = Counter(run_objs).most_common(1)[0][0] if run_objs else "unknown"
                conf = float(np.nanmean(np.exp(vlp[a:b + 1, lab]))) if not np.isnan(vlp[a:b + 1, lab]).all() else None
                s = max(0.0, times[a] - step / 2)
                e = min(meta.duration_s, times[b] + step / 2)
                if e - s >= c.min_segment_s:
                    actions.append(Action(clip=meta.name, start_s=round(s, 2), end_s=round(e, 2), verb=c.verbs[lab],
                                          object=obj, hand=side, confidence=round(conf, 3) if conf else None,
                                          source=self.name))
        actions = carve_pick_place(actions, contacts, c.pick_place_s, c.min_segment_s)
        return merge_both(actions, c.both_iou)
