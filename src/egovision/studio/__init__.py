"""Studio: upload an industrial video in the browser, analyse it, watch events arrive live.

Backend is the source of truth: OpenCV samples timestamped frames, Gemini returns structured JSON per
batch (or, without an API, FORGE's offline labeller produces the events), Pydantic validates it, SQLite
stores it, Python computes every worker metric, and CSV / JSON reports are generated from the database
into `outputs/videos/<video_id>/reports/`. The browser only uploads, displays and downloads. Existing FORGE
results can be imported without any processing. Start with `egovision studio` (http://127.0.0.1:8010).
"""
