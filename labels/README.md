# Labels (our own, not the organisers')

The hackathon data has no labels (PDF p.1) and no official metric (SPEC.md §3). These files are
**our own** validation labels, used only to measure our system. Every score computed from them must
carry this provenance.

## Provenance (read before quoting any number)

| File | What | Who made it | Verified by a person? |
|---|---|---|---|
| `mytronlabs-000060-016.actions.csv` | Action segments, 0–180 s | Drafted by Claude Code from 1 frame/s contact sheets | **No — pending team check** |
| `MTL-SKL00089-007.actions.csv` | Action segments, 0–60 s only (see `.meta.json`) | Drafted by Claude Code from 1 frame/s sheets + zoomed frames | **No — pending team check** |
| `dev/mytronlabs-000060-010.actions.csv` | Action segments, 0–60 s — **dev split, used for tuning decisions** | Drafted by Claude Code from 1 frame/s sheets, before any prediction for this clip was looked at | **No — pending team check** |
| `hands_judged.csv` (when present) | Per-frame hand judgements from `scripts/make_judging_sheets.py` | Team | — |

Until a teammate has checked the drafts against the video, scores from them are **indicative only**
and RESULTS.md says so.

## Format

`<clip>.actions.csv` uses the PS3 columns: `clip,start_s,end_s,verb,object,hand`, verbs from the fixed
list in `src/egovision/config.py`. `<clip>.meta.json` holds `labelled_until_s`; the evaluator only scores
that window. Actions of different hands may overlap; `both` covers both hand lanes.

## Splits

- Held out for reporting (never used to tune anything): `mytronlabs-000060-016`, `MTL-SKL00089-007`.
- Tuning (dev): `dev/mytronlabs-000060-010` (0–60 s). Score with `egovision eval --labels labels/dev`.

## How to check a draft

1. Open the clip in the dashboard (`egovision ui`), Actions tab next to the video.
2. For each row in the label file, watch the segment; fix verb/object/hand/times in the CSV.
3. Add missing actions, delete wrong ones, then change "pending team check" above to your name and date.
