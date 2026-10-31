"""Streamlit demo: pick or upload a clip, run the pipeline, watch the overlay, browse the timeline,
download the PS3 file. Launch with `egovision ui` (or `streamlit run src/egovision/ui.py`)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from egovision.config import Config
from egovision.export import load_result
from egovision.render import verb_colors

RAW = Path("data/raw")
UPLOADS = Path("data/uploads")
OUT = Path("outputs")
LABELS = Path("labels")

st.set_page_config(page_title="egovision · PS3 task-step timeline", layout="wide")


def _hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def pick_video() -> Path | None:
    st.sidebar.header("1 · Video")
    clips = sorted(RAW.glob("*.mp4")) + sorted(UPLOADS.glob("*.mp4"))
    up = st.sidebar.file_uploader("Upload an .mp4", type=["mp4"])
    if up is not None:
        UPLOADS.mkdir(parents=True, exist_ok=True)
        dst = UPLOADS / Path(up.name).name
        if not dst.is_file():
            dst.write_bytes(up.getbuffer())
        return dst
    if not clips:
        st.sidebar.warning(f"No videos in {RAW}/ — upload one.")
        return None
    done = {p.parent.name for p in OUT.glob("*/result.json")}
    labels = [f"{'✓ ' if c.stem in done else ''}{c.name}" for c in clips]
    i = st.sidebar.selectbox("…or pick a sample clip (✓ = results exist)", range(len(clips)), format_func=labels.__getitem__)
    return clips[i]


def run_controls(video: Path) -> None:
    st.sidebar.header("2 · Run")
    max_s = st.sidebar.number_input("Process first N seconds (0 = whole clip)", 0, 3600, 30, 10)
    stride = st.sidebar.select_slider("Hand-model frame stride", [1, 2, 3, 4], value=2)
    objects = st.sidebar.checkbox("Object detection + contacts", value=True)
    if st.sidebar.button("▶ Run pipeline", type="primary"):
        from egovision.pipeline import run_video

        ov = [f"ingest.stride={stride}", f"objects.enabled={str(objects).lower()}"]
        if max_s:
            ov.append(f"ingest.max_seconds={max_s}")
        bar = st.progress(0.0, text="starting…")
        t0 = time.time()
        try:
            run_video(video, OUT, Config.load(None, ov), render=True,
                      progress=lambda stage, f: bar.progress(min(max(f, 0.0), 1.0), text=f"{stage} {f:.0%}"))
        except Exception as e:  # noqa: BLE001 - show any pipeline failure in the UI
            st.error(f"Pipeline failed: {e}")
            return
        bar.progress(1.0, text=f"done in {time.time() - t0:.0f}s")


def show_results(video: Path) -> None:
    out_dir = OUT / video.stem
    if not (out_dir / "result.json").is_file():
        st.info("No results for this clip yet — press **Run pipeline** in the sidebar.")
        return
    r = load_result(out_dir)
    meta, actions, stats = r["meta"], r["actions"], r["run"].get("stats", {})
    timings = r["run"].get("timings", {})
    st.subheader(f"{meta.name}  ·  {meta.duration_s:.0f}s processed")
    c = st.columns(6)
    c[0].metric("Frames with a hand", f"{stats.get('all_frames_with_hand_pct', '–')}%")
    c[1].metric("Both hands (processed)", f"{stats.get('processed_with_two_hands_pct', '–')}%")
    c[2].metric("Hand model speed", f"{timings.get('hands_fps', '–')} fps")
    c[3].metric("Actions", len(actions))
    c[4].metric("Contact segments", len(r["contacts"]))
    c[5].metric("Device", stats.get("device", "–"))

    left, right = st.columns([3, 2])
    with left:
        start = st.session_state.get("seek", 0)
        overlay = out_dir / "overlay.mp4"
        if overlay.is_file():
            st.video(str(overlay), start_time=int(start))
        else:
            st.warning("overlay.mp4 missing (run with rendering on).")
    with right:
        st.markdown("**Step timeline** (click a row, then *Jump*)")
        df = pd.DataFrame([a.model_dump(include={"start_s", "end_s", "hand", "verb", "object", "confidence"})
                           for a in actions])
        if not df.empty:
            sel = st.dataframe(df, height=360, use_container_width=True, on_select="rerun", selection_mode="single-row")
            rows = sel.selection.rows if sel and hasattr(sel, "selection") else []
            if rows and st.button(f"Jump to {df.iloc[rows[0]]['start_s']:.1f}s"):
                st.session_state["seek"] = float(df.iloc[rows[0]]["start_s"])
                st.rerun()
    if actions:
        import altair as alt

        colors = verb_colors(Config().steps.verbs)
        verbs = sorted({a.verb for a in actions})
        tl = pd.DataFrame([{"lane": h, "start": a.start_s, "end": a.end_s, "verb": a.verb, "object": a.object}
                           for a in actions for h in ({"left": ["left"], "right": ["right"]}.get(a.hand, ["left", "right"]))])
        chart = alt.Chart(tl).mark_bar(height=22).encode(
            x=alt.X("start:Q", title="seconds", scale=alt.Scale(domain=[0, meta.duration_s])), x2="end:Q",
            y=alt.Y("lane:N", sort=["left", "right"], title=None),
            color=alt.Color("verb:N", scale=alt.Scale(domain=verbs, range=[_hex(colors[v]) for v in verbs])),
            tooltip=["verb", "object", "lane", "start", "end"]).properties(height=110)
        st.altair_chart(chart, use_container_width=True)

    st.markdown("**Downloads**")
    d = st.columns(4)
    for col, name, mime in [(d[0], "actions.csv", "text/csv"), (d[1], "contacts.csv", "text/csv"),
                            (d[2], "hands.jsonl", "application/jsonl"), (d[3], "result.json", "application/json")]:
        f = out_dir / name
        if f.is_file():
            col.download_button(f"⬇ {name}", f.read_bytes(), file_name=f"{meta.name}_{name}", mime=mime,
                                type="primary" if name == "actions.csv" else "secondary")
    with st.expander("Contacts (grab / release timeline)"):
        st.dataframe(pd.DataFrame([c.model_dump() for c in r["contacts"]]), use_container_width=True)
    gt = LABELS / f"{meta.name}.actions.csv"
    if gt.is_file():
        from egovision.evaluate import eval_steps
        from egovision.export import read_actions_csv

        with st.expander("Evaluation against our labels for this clip", expanded=True):
            st.json(eval_steps(read_actions_csv(out_dir / "actions.csv"), read_actions_csv(gt), meta.duration_s))
    with st.expander("Run details (timings, stats, config)"):
        st.json(json.loads(json.dumps(r["run"])))


st.title("egovision — task-step timeline from head-camera factory video")
st.caption("PS3 · hands (MediaPipe + tracking) → objects (open-vocabulary) → contacts → steps (exemplar matching)")
v = pick_video()
if v is not None:
    run_controls(v)
    show_results(v)
