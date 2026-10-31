# Pipeline

The full FORGE PS3 pipeline, stages 0–14. The diagram renders on GitHub; paste it into
https://mermaid.live to export a PNG/SVG for the slides.

Colours: teal = automated stages, purple = privacy and data handling, gray = inputs and people.

```mermaid
flowchart TD
    RAW[/"Factory MP4<br/>original, local only"/]
    S0["0 · Ingest + QC<br/>decode, blur score, lighting score"]
    S1["1 · Privacy sanitization<br/>faces, company names, screens, documents<br/>hands kept visible"]
    S2["2 · Worker profile<br/>expected hands, gloves, prosthetic"]
    S3["3 · Hand tracking<br/>detect, track IDs, side fix, recovery, smoothing"]
    S4["4 · Camera-origin coordinates<br/>x, y, zone, speed, visibility → signals.csv"]
    S5["5 · VLM timeline (Gemini)<br/>pass 1 full clip, pass 2 targeted windows<br/>sanitized copy only"]
    S6["6 · Schema validation<br/>Pydantic, fixed verb list, times in range"]
    S7["7 · Hand-grounded verification<br/>verified / corrected / unverified / rejected"]
    S8["8 · Step order + completeness<br/>SOP alignment, motion-valley boundary snapping,<br/>missing / out-of-order flags"]
    S9["9 · Human-in-the-loop review<br/>uncertainty-sorted queue: approve, edit, split, merge, reject"]
    S10["10 · Trim + compress<br/>cut idle or keep-lite, time map"]
    S11["11 · Analytics<br/>step durations, idle time, work zones, cycles"]
    S12["12 · Render + dashboard<br/>sanitized video, overlays, timeline, downloads"]
    S13["13 · Evaluation<br/>before vs after, measured on reviewed labels"]
    S14["14 · Data lifecycle<br/>delete every copy at event end"]

    RAW --> S0
    S0 --> S1 & S2 & S3
    S1 -->|sanitized MP4| S5
    S2 --> S3
    S3 --> S4
    S5 --> S6 --> S7
    S4 -->|hand evidence| S7
    S7 --> S8 --> S9
    S9 --> S10 & S11
    S10 --> S12
    S11 --> S12
    S11 --> S13
    S12 --> S14
    S13 --> S14

    OUT[["actions.csv · actions.json · hands.jsonl · signals.csv"]]
    S9 --> OUT --> S12

    classDef auto fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef priv fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef human fill:#F1EFE8,stroke:#5F5E5A,color:#444441
    class S0,S3,S4,S5,S6,S7,S8,S11,S12,S13 auto
    class S1,S10,S14 priv
    class RAW,S2,S9,OUT human
```

## Stage summary

| # | Stage | Input → output |
|---|---|---|
| 0 | Ingest + QC | MP4 → frames, `qc.json` |
| 1 | Privacy sanitization | frames + detections → `data/sanitized/<clip>.mp4`, `privacy_log.json` |
| 2 | Worker profile | team-declared YAML → expected manipulators |
| 3 | Hand tracking | original frames → `hands.jsonl` |
| 4 | Camera-origin coordinates | hand tracks → `signals.csv` |
| 5 | VLM timeline | sanitized MP4 → `raw_pass1.json`, `raw_pass2.json` |
| 6 | Schema validation | raw responses → `actions_valid.json` |
| 7 | Hand-grounded verification | actions + signals → status per action |
| 8 | Step order + completeness | verified actions → `actions_refined.json` |
| 9 | Human review | refined actions → `review.json` |
| 10 | Trim + compress | reviewed video → `data/trimmed/<clip>_<mode>.mp4` (`cut`: shorter; `keep_lite`: same length), `trim_map.json` (plan, time map), `actions_trimmed.csv` (cut mode) |
| 11 | Analytics | actions + signals → durations, zones, cycles |
| 12 | Render + dashboard | everything → JS dashboard, annotated MP4 |
| 13 | Evaluation | outputs + reviewed labels → before/after metrics |
| 14 | Data lifecycle | all copies → deleted at event end |

Build status per stage is tracked in `docs/PLAN.md`, not here.

## Privacy guarantees in code
- The dashboard only streams the sanitized copy (`/media/<clip>/sanitized.mp4`); originals are never served.
- The Gemini backend refuses to upload an unsanitized clip unless `steps.allow_unsanitized_upload=true` is set deliberately.
- Overlay and demo renders draw on the sanitized copy.
- Videos are written atomically, so an interrupted run never leaves a half-written file.
