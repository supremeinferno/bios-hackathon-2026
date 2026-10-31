# Errors and fixes log

Every mistake, failure or wrong assumption we hit while building this project, how we noticed
it, why it happened, and how it was fixed. It is kept up to date as we work, so the
"what failed" slide can be taken straight from here. Newest entries are at the bottom.

**Legend.** Type: **Data** (download/data handling), **Env** (installs, libraries), **Model**
(a model or method that did not work on our footage), **Method** (wrong assumption or
measurement), **Repo** (git/data-rule hygiene), **Code** (bugs). Status: ✅ fixed · ⚠️ mitigated
(a limit remains) · ⏳ open.

## Summary

| # | Phase | Type | What went wrong | Fix | Status |
|---|---|---|---|---|---|
| 1 | Phase 0 | Data | 28 of 40 clips downloaded with a stray character in the file name; 12 failed | Strip Windows line endings from the download list, rename, re-download | ✅ |
| 2 | Phase 0 | Data | 4 clips silently truncated (download "succeeded") | Check every file's size against the server's `Content-Length`, re-fetch mismatches | ✅ |
| 3 | Phase 0 | Method | Blur metric said 0% of frames were blurry, though blur was clearly visible | Use a per-clip relative sharpness threshold instead of a fixed one | ✅ |
| 4 | Phase 0/M1 | Env | `pip` reported "no matching distribution for scipy" | Transient network failure; retried with `--retries` | ✅ |
| 5 | M1 | Model | Every left/right hand label was reversed | Measured it; turned off the PDF's suggested handedness swap | ✅ |
| 6 | M1 | Repo | First commit included pipeline outputs (derived from company footage) and build files | Amended the unpushed commit; added `outputs/`, `*.egg-info/` to `.gitignore` | ✅ |
| 7 | Block 1 | Method | Video decoding was the bottleneck (OpenCV 106 fps) | Switched ingest to PyAV (357 fps) | ✅ |
| 8 | Block 1 | Env | OWLv2 post-processing call crashed after upgrading to transformers 5 | Use the renamed API `post_process_grounded_object_detection` | ✅ |
| 9 | Block 1 | Env | `torch_dtype` deprecated in transformers 5 | Use `dtype=` | ✅ |
| 10 | Block 1 | Env | SigLIP failed to load: tokenizer needs SentencePiece | Installed `sentencepiece` + `protobuf` | ✅ |
| 11 | Block 1 | Env | SigLIP embeddings crashed: model returns an output object, not a tensor (transformers 5) | Small `_tensor()` helper that accepts both | ✅ |
| 12 | Block 1 | Model | Grounding DINO attached object names to the wrong boxes | Measured both detectors; chose OWLv2 (3.5x faster, cleaner labels) | ✅ |
| 13 | Block 1 | Method | Verb list had no verb for operating a hoist remote | Added `press` (it is in the PDF's own example verbs) before scoring | ✅ |
| 14 | Block 2 | Method | Action labels are ambiguous at 1 frame/s on fine bench work | Labelled only the confident window (0–60 s) and score only that window | ⚠️ |
| 15 | Stage 1 | Model | MediaPipe face detector put "face" boxes on hands and missed distant faces | Blur heads from OWLv2 **person** boxes instead (+ head/face prompts) | ✅ |
| 16 | Stage 1 | Env | Download of a second face model failed (connection reset) | Not retried; redesigned to use the detector we already had | ✅ |
| 17 | Stage 1 | Code | Shell patch command failed on quoting (nothing was changed) | Wrote the patch as a Python file and ran that | ✅ |
| 18 | Stage 1 | Code | Overlay canvas was sized 44 px shorter than the video | Canvas covers the full video area (`height: 100%`) | ✅ |
| 19 | Stage 1 | Method | Privacy blur covered too much: 99% of frames, 15% of each frame on average; wearer's own knees blurred as "heads" | Drop heads low in the frame; smaller padding and head band; ellipses; shorter hold | ✅ |
| 20 | Stage 1 | Code | A run stopped mid-write left a corrupt sanitized video that the UI could have served | Videos are now written to a temp file and renamed only when complete | ✅ |
| 21 | Stage 1 | Model | Part of a company sign stays readable at the frame edge; false "person" boxes blur wooden panels | Open: flagged for the human privacy check | ⏳ |
| 22 | Stage 1 | Model | When the wearer squats (SGF clips), their own knees and forearms mid-frame get pixelated; only the detected hand boxes stay sharp | Open: needs a better wearer-body filter | ⏳ |
| 23 | Stage 1 | Method | Work objects (tools, printed templates, tape rolls, metal parts) were pixelated | Privacy limited to faces and company names; text/document/screen/monitor prompts removed | ✅ |
| 24 | Stage 5 | Env | Gemini returned 503 "high demand" on every retry, and the failure crashed the whole run (hands, contacts lost) | Offline labeller always runs; Gemini is a layer on top that falls back to it | ✅ |
| 25 | Stage 5 | Code | One Gemini request hung for 27 minutes with no timeout | 5-minute request timeout, then the next model | ✅ |
| 26 | Stage 5 | Env | `gemini-2.5-flash` is listed for our key but returns 404 "not available to new users"; retries were wasted on it | Model list set to the API's recommendation (3.8 → 3.7 → 3.5 → latest); 4xx errors are not retried | ✅ |
| 27 | Stage 7 | Code | Verification checked hand presence before correcting the hand, so corrected actions came out "unverified" | Attribution first, then presence on the corrected hand (caught by a unit test) | ✅ |
| 28 | Stage 8 | Code | Boundary snapping made Gemini's timeline worse (F1@50 45.9 → 26.2, boundary error 0.80 → 1.55 s) | Snap only to samples where the claimed hand is visible with a real speed; otherwise leave the boundary | ✅ |
| 29 | Stage 5 | Env | Whole 3-minute clips: 503 on every Gemini model for over an hour; 10-second clips answered in 3.6 s | Gemini is called on 30 s windows (+ pass 2 around uncertain actions) | ✅ |
| 30 | Stage 3 | Model | Missed-hand recovery (CLAHE retry) found conveyor rollers, a foot and forearms as "hands", for ≤1 point more presence at half the speed | Measured on 4 clips; recovery off by default | ✅ |
| 31 | Stage 3 | Method | The coworker-hand filter dropped the wearer's OWN hand (fingers curled down near the camera) | Measured + visually checked; filter off by default | ✅ |
| 32 | Stage 5 | Env | Free-tier quota for `gemini-3.5-flash` ran out during the 40-clip batch (429); every window then wasted ~20 s retrying it | Quota errors are not retried and the model is skipped for the rest of the run; batch clips are labelled by the fallback `gemini-flash-lite-latest`, recorded per clip | ⚠️ |

## Details

### 1 · Download list had Windows line endings → broken file names
- **Symptom.** After downloading the 40 sample clips, only 28 files existed and their names showed a
  stray character (`MTL-SGF00106-021.mp4_.part`); `mv` failed for all of them; 12 clips were missing.
- **Root cause.** The file list (Drive file ID + name per line) was written by Python on Windows with
  `\r\n` line endings. The download script read the `\r` as part of the file name.
- **Fix.** Stripped `\r` from the list, renamed the 28 files programmatically, re-ran the download for the
  missing 12.
- **Lesson.** On Windows, normalise line endings before feeding text into shell loops.

### 2 · Silent truncation of 4 clips
- **Symptom.** Four `MTL-SKL00089` clips were 18–95 MB instead of ~108 MB, but the download reported success.
- **How we caught it.** We checked every local file size against the server's `Content-Length` header
  instead of trusting the exit code.
- **Fix.** Deleted and re-downloaded the four; all 40 now match the server byte-for-byte.
- **Lesson.** Always verify downloads against an independent size or checksum.

### 3 · Fixed blur threshold reported "0% blurry"
- **Symptom.** A Laplacian-variance threshold of 60 classified 0% of sampled frames as blurry on every clip,
  but contact sheets clearly showed motion-blurred frames.
- **Root cause.** Whole-frame sharpness is dominated by texture (sawdust, granite); median values ranged
  from 1.8k to 8.4k between sources, so a fixed threshold is meaningless.
- **Fix.** Count a frame as blurry when it is below 35% of that clip's median sharpness. This flags 7–20%
  of frames in the woodworking clips and 0–3% elsewhere, matching visual inspection. (DECISIONS D2)
- **Limit.** Whole-frame sharpness still misses fast hand motion over a static background.

### 4 · `pip` "no matching distribution for scipy"
- **Symptom.** `ERROR: Could not find a version that satisfies the requirement scipy (from versions: none)`.
- **Root cause.** A transient network failure; `pip download scipy` worked seconds later.
- **Fix.** Re-ran the install with `--retries 8 --timeout 60`.

### 5 · Handedness labels were all reversed
- **Symptom.** In the overlay video every hand on the wearer's left arm was labelled "right" and vice versa.
- **How we caught it.** A built-in sanity metric: only **20.6%** of "left" hands had their wrist in the left
  half of the image (expected: most of them). Then a visual check of 4 frames with both hands: 4/4 reversed.
- **Root cause.** The hackathon PDF (p.11) warns that MediaPipe labels hands as if the image were a mirrored
  selfie and suggests swapping. With MediaPipe 1.2.0 on our non-mirrored head-camera video, the raw labels
  are already correct, so the swap made them wrong.
- **Fix.** `hands.swap_handedness = false`. Every run still reports `side_matches_image_half_pct` as a
  regression check. (DECISIONS D4)
- **Lesson.** Measure advice on your own data before applying it.

### 6 · Derived data committed to git
- **Symptom.** The M1 commit included `outputs/` (hand coordinates and timelines computed from the
  company's footage) and `egg-info` build files.
- **Why it matters.** The repo must be public, and the data-use rules forbid publishing anything from the
  footage. Breaking them means disqualification.
- **Fix.** Removed them from the commit (amended before anything was pushed) and added `outputs/` and
  `*.egg-info/` to `.gitignore`. Verified that no video, frame, image, `.jsonl` or `outputs/` file is tracked.

### 7 · Decoding was the bottleneck
- **Symptom.** Full-clip ingest took ~58 s of a ~5 min run.
- **Measurement.** Same clip, decode + resize to 960×540: OpenCV 106 fps vs PyAV (threaded decode,
  scaling inside the decoder) 357 fps.
- **Fix.** Ingest rewritten on PyAV; ingest for a 3-minute clip dropped to ~15 s. (DECISIONS D5)

### 8 – 11 · transformers 5 API changes
| # | Error | Fix |
|---|---|---|
| 8 | `AttributeError: 'Owlv2Processor' object has no attribute 'post_process_object_detection'` | Call `post_process_grounded_object_detection` (same arguments) |
| 9 | Warning: `torch_dtype` is deprecated | Pass `dtype=` to `from_pretrained` |
| 10 | `ImportError: SiglipTokenizer requires the SentencePiece library` | `pip install sentencepiece protobuf` |
| 11 | `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'float'` | `get_image_features` / `get_text_features` now return an output object; take `.pooler_output` when present |

- **Root cause.** The newest transformers (5.22) changed several APIs that most examples online still use.
- **How we caught them.** The 10-second smoke test runs the whole pipeline on a real clip before commits.

### 12 · Grounding DINO mislabelled boxes
- **Symptom.** On the same 8 keyframes, Grounding DINO put "hand truck" and "metal bar" boxes on unrelated
  regions, and it was 3.5× slower than OWLv2 (2.1 vs 7.4 images/s on the RTX 3050).
- **Root cause.** Grounding DINO maps boxes to words in one long joined prompt; similar phrases bleed into
  each other. OWLv2 scores each prompt separately.
- **Fix.** OWLv2 is the default detector (Grounding DINO is still selectable). Threshold raised 0.15 → 0.2
  to cut low-score noise. (DECISIONS D8)
- **Limit.** OWLv2 rarely detects very large surfaces (granite slab scored 0.21).

### 13 · Missing verb `press`
- **Symptom.** While labelling, the stone-plant worker repeatedly operated a hand-held hoist remote, and no
  verb in our 16-verb list fit.
- **Fix.** Added `press`, which the PDF lists among example verbs. This was done before any predictions were scored.
  (DECISIONS D6)

### 14 · Ambiguous labels on fine bench work ⚠️
- **Symptom.** In `MTL-SKL00089-007`, at 1 frame/s it is unclear when "wrap", "tape" and "adjust" start and end.
- **Fix.** Labelled only the first 60 s (where we could zoom in to confirm the task cycle), stored the
  window in `labels/<clip>.meta.json`, and the evaluator scores only that window.
- **Still open.** All labels were drafted by Claude Code from frame sheets and are **not human-verified**;
  a teammate must check them before we quote scores as final.

### 15 · MediaPipe face detector unusable on our footage
- **Symptom.** Benchmark on 10 frames with known faces, a sign, a document and screens:
  MediaPipe Face (short-range) found the one close face but missed every distant face, and put "face"
  boxes on **hands and arms** in 3 frames. Blurring hands would destroy our tracking evidence.
- **Root cause.** The short-range model is built for selfie-distance faces (< 2 m), not fisheye factory
  scenes.
- **Fix.** Use OWLv2 (already loaded for objects) with privacy prompts. "person" boxes found nearly every
  coworker, including small and distant ones; we blur the head band (top 35%) of each person box, plus
  "human head"/"human face", signage, logos, text, screens and documents. Person boxes that are the
  wearer's own arms are filtered out, and the wearer's hand boxes are cut out of the blur mask.
  (DECISIONS D9)
- **Limit.** No dedicated OCR yet; small printed text relies on OWLv2's "printed text" prompt.

### 16 · Face model download failed
- **Symptom.** `curl: (35) Recv failure: Connection was reset` while fetching a second face model from GitHub.
- **Fix.** We did not keep retrying. We changed the design to need no new downloads (see #15).

### 17 · Shell quoting broke a patch command
- **Symptom.** `bash: unexpected EOF while looking for matching ''`; nothing was modified.
- **Fix.** Wrote the code edits as a Python script file and ran it; verified with the unit tests (30 passed).

### 18 · Overlay canvas misaligned with the video
- **Symptom.** In the new dashboard CSS the drawing canvas was `100% − 44px` tall, so hand boxes would
  be drawn slightly above the hands.
- **Fix.** The canvas now covers the full video area. The video's native controls stay clickable because the
  canvas ignores the mouse.

### 19 · Over-blurring in the first privacy pass
- **Symptom.** On `MTL-SGF00106-021`, 99.1% of frames had pixelation covering 15.3% of the frame on average.
  Large wooden panels were pixelated, and the wearer's own knee and the area around the hand (bottom of
  the frame) were pixelated as "heads".
- **How we measured it.** Per-category breakdown of the privacy log: 1,835 head regions on 540 keyframes;
  15% of them started in the bottom 30% of the frame.
- **Root cause.** (a) In head-camera video, the wearer's own knees, feet and body sit at the bottom of the
  frame and look like heads to the detector. (b) Generous geometry: 20% padding, head band = top 35% of a
  person box, rectangles, ±0.5 s hold that smears boxes while the camera moves.
- **Fix.** Drop head/face regions whose top edge is below 70% of the frame height; padding 20% → 10%; head
  band 35% → 25%; heads and faces pixelated as ellipses; hold ±0.5 s → ±0.35 s (keyframes are 0.33 s
  apart, so coverage stays continuous).
- **Result (first 40 s, same clips).** Mean pixelated area when blurring: SGF-021 **15.3% → 3.1%**,
  000060-006 **12.5% → 8.7%** (mostly the company sign). Coworker faces in the checked frames stay covered.
  (DECISIONS D10)

### 20 · Half-written sanitized video
- **Symptom.** `Invalid data found when processing input: data/sanitized/mytronlabs-000003-002.mp4`.
- **Root cause.** The run was stopped while the video was being written; the file was written in place, so
  a truncated MP4 without its index was left behind, and the dashboard would have tried to stream it.
- **Fix.** `write_video` writes to `.<name>.partial.mp4` and renames it only after the encoder is flushed;
  on any error or interrupt the partial file is deleted. Unit test `test_interrupted_write_leaves_no_file`.

### 21 · Remaining privacy misses (open) ⏳
- **What.** In `mytronlabs-000060-006` at 25 s the main sign is pixelated, but a fragment of its text at the
  right edge stays readable. In SGF-021 a wooden panel at the top is still pixelated by false "person" boxes.
- **Plan.** Human privacy check on sampled frames (FORGE Stage 9); add OCR-based text detection if misses
  persist. Until then, sanitized clips stay private like the originals.

### 22 · Wearer's own body pixelated mid-frame (open) ⏳
- **Symptom.** First dashboard check on `MTL-SGF00106-023` (30 s run from the UI): the worker squats, so the
  knees, feet and forearms sit in the middle of the frame. They are pixelated as "person"/"head", and only
  the two detected hand boxes stay sharp.
- **Why the earlier fix missed it.** The bottom-of-frame rule (#19) only drops heads that start in the
  lower 30%; a squatting wearer's knees start higher.
- **Plan.** Treat person boxes that overlap a wearer hand track or the wearer's forearm direction as the
  wearer, not just boxes touching the bottom edge; verify on SGF-023 and SGF-021.

### 23 · Workers' tools and parts were pixelated
- **Symptom.** Reported from the dashboard: objects the workers were using were blurred, which hides
  exactly what the task-step timeline is about.
- **Root cause.** The `printed text`, `paper document`, `phone screen` and `computer monitor` prompts
  matched work objects: printed plywood templates, tape rolls, labelled boxes, shiny metal parts.
- **Fix.** Blur only faces and company names: prompts reduced to `person` (head band), `head`, `face`,
  `sign`, `logo`. On SGF-023 (first 30 s) the nail gun, template and planer are now sharp. (DECISIONS D11)
- **Remaining.** Small false "head" blobs still appear occasionally (one on a planer guard, one beside the
  worker's leg in the checked frames); see #22 for the related wearer-body issue. Company names on small
  items are not caught without OCR.

### 24 · Gemini overload crashed the whole run
- **Symptom.** First Gemini run on `mytronlabs-000060-016`: `503 UNAVAILABLE: This model is currently
  experiencing high demand` on all retries; the exception propagated and the run ended with only
  `privacy_log.json` written. Hand tracks, contacts and the offline timeline were lost.
- **Root cause.** The VLM was the only step labeller in the run, and its error was not caught.
- **Fix.** The offline matcher now always runs (it also provides the embeddings and Stage 7's comparison).
  Gemini runs on top; if every model fails, the run logs the error in `run.json` (`vlm_error`) and keeps
  the offline timeline. Verified on the second run: the pipeline finished and wrote all outputs.
- **Data rule check.** Gemini file storage was listed after each failure: 0 files left (uploads deleted in `finally`).

### 25 · 27-minute hang
- **Symptom.** A retry started at 19:25:46 and only failed at 19:53:06 ("Server disconnected").
- **Fix.** `HttpOptions(timeout=300 s)` on the client (`steps.request_timeout_s`).

### 26 · Listed model that cannot be used
- **Symptom.** `404 NOT_FOUND: models/gemini-2.5-flash is no longer available to new users`, even though
  `models.list()` returns it for our key. It was retried 3 times anyway.
- **Fix.** Main model `gemini-3.8-flash` (the model the 404 message itself recommends), fallbacks
  3.7 → 3.5 → `gemini-flash-latest`; client errors (4xx except 429 rate limit) are not retried. (DECISIONS D13)
- **Lesson.** "Listed" does not mean "usable"; read the error body.

### 27 · Verification order bug
- **Symptom.** Unit test `test_hand_corrected_when_other_hand_did_it`: hand correctly changed to `left`,
  but status `unverified` instead of `corrected`.
- **Root cause.** The presence check ran on the originally claimed (wrong) hand before the correction.
- **Fix.** Attribution first, then presence and motion checks on the corrected hand.

### 28 · Boundary snapping pulled boundaries into "hand not visible" gaps
- **Symptom.** First scored run on held-out `mytronlabs-000060-016`: Gemini raw F1@50 = 45.9 and boundary
  error 0.80 s; after Stage 7–8 the same timeline scored F1@50 = 26.2 and 1.55 s.
- **Root cause.** `signals.csv` leaves speed empty when a hand is not visible, and the snapping code read
  empty as 0, i.e. "perfectly still". On a clip where hands are visible in only ~15% of frames, the
  "slowest moment" was almost always a moment with no hand at all.
- **Fix.** Snap only to samples where the claimed hand(s) have a real speed, and only when that holds for
  ≥60% of the ±1.2 s window; otherwise keep the boundary. Unit test
  `test_no_snap_into_frames_where_the_hand_is_invisible`.
- **Result (same held-out clip).** Snapping is now neutral (F1@50 42.6 with or without snapping; boundary
  error 0.83 s vs 0.80 s raw). The remaining drop vs raw comes from Stage 7's hand correction on this clip
  (hand accuracy 66.7 → 55.6), whose evidence is weak when the tracker misses most hands. The on/off
  decision is made on the dev clip, not on this held-out one (DECISIONS D15).
- **Lesson.** "Missing" is not "zero". A refinement stage needs its own before/after measurement.

### 29 · Gemini refuses whole clips, answers short windows
- **Symptom.** Two full runs: every model (3.8, 3.7, 3.5, flash-latest) returned 503 "high demand" for a
  3-minute clip. A tiny text request and a 10-second clip both answered within seconds on `gemini-3.5-flash`.
- **Fix.** 30-second windows (one decode pass cuts them all), times shifted back to clip time, actions
  stitched across window edges, pass 2 re-asks windows around uncertain actions at 4 fps. First windowed
  run: 6 windows → 33 actions, one transient 503 recovered by retry. (DECISIONS D14)

### 30 · Missed-hand recovery mostly added false hands
- **Measured** (first 60 s of 4 clips, `scripts/measure_hands.py`): frames with a hand 92.0 → 93.0% (SKL-005),
  11.8 → 11.8% (000060-016), 56.3 → 55.6% (SGF-036), 90.4 → 90.4% (000003-002); hand-model speed
  35–50 fps → 18–22 fps.
- **Visual check of recovered hands:** stone plant: 7 of 12 were pink conveyor rollers, 2 forearms;
  bench clip: one was the wearer's foot, two were forearms.
- **Decision.** `hands.recover = false` (kept as an option). (DECISIONS D12)

### 31 · Coworker filter removed the wearer's own hand
- **Rule.** Drop a hand whose fingers point down toward the camera from the upper part of the frame
  (a coworker facing the wearer).
- **Visual check.** On `mytronlabs-000003-002` all 12 dropped hands were the wearer's own hand, held close to
  the lens with fingers curled (wristband and red shirt visible). On 000060-016 the 5 drops were rollers.
- **Decision.** `hands.filter_other_person = false`. Coworker hands remain a known limitation.
- **Lesson.** A geometric rule that "makes sense" for egocentric video was wrong on our footage; measure
  and look before switching anything on.

### 32 · Free-tier quota ran out mid-batch ⚠️
- **Symptom.** `429 RESOURCE_EXHAUSTED: You exceeded your current quota` for `gemini-3.5-flash` from the
  first batch clips on (earlier dev/held-out runs had used part of the free daily quota).
- **What happened.** The model fallback worked: windows were answered by `gemini-flash-lite-latest`. But each
  window first retried the exhausted model 3 times (~20 s wasted per window).
- **Fix.** A 429 that mentions "quota" is not retried, and that model is skipped for the rest of the run.
  `run.json` now records `vlm_windows_by_model`, so every clip says which model labelled it.
- **Still open.** The measured quality (RESULTS §1) is for `gemini-3.5-flash`; most batch clips were labelled by
  `gemini-flash-lite-latest`, which we have not scored separately. Re-running those clips after the daily
  quota resets would align them (responses are cached per window, so only missing windows are re-asked).
