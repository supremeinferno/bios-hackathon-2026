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
