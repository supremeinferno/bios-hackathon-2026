"""Command line: `egovision run | batch | eval | ui`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from egovision.config import Config

app = typer.Typer(add_completion=False, help="Egocentric factory video -> hand tracks, contacts and a task-step timeline.")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="YAML config file.")]  # noqa: UP007
SetOpt = Annotated[list[str], typer.Option("--set", "-s", help="Override, e.g. --set ingest.stride=3")]
MaxSecOpt = Annotated[float | None, typer.Option(help="Only process the first N seconds")]  # noqa: UP007


@app.callback()
def _main() -> None:
    # Read GEMINI_API_KEY etc. from .env if present; existing environment variables win.
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load(config: Path | None, overrides: list[str], max_seconds: float | None) -> Config:
    ov = list(overrides)
    if max_seconds is not None:
        ov.append(f"ingest.max_seconds={max_seconds}")
    return Config.load(config, ov)


@app.command()
def run(
    video: Annotated[Path, typer.Option("--video", "-v", exists=True, dir_okay=False, help="Input .mp4")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output root; results go to OUT/<clip>/")] = Path("outputs"),
    config: ConfigOpt = None,
    set_: SetOpt = [],  # noqa: B006 - typer needs a literal default
    render: Annotated[bool, typer.Option(help="Write overlay.mp4")] = True,
    max_seconds: MaxSecOpt = None,
    verbose: bool = False,
) -> None:
    """Process one clip."""
    _setup_logging(verbose)
    from egovision.pipeline import run_video

    res = run_video(video, out, _load(config, set_, max_seconds), render=render)
    typer.echo(json.dumps({"out_dir": str(res.out_dir), "timings": res.timings, "stats": res.stats}, indent=1))


@app.command()
def batch(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)] = Path("data/raw"),
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("outputs"),
    config: ConfigOpt = None,
    set_: SetOpt = [],  # noqa: B006
    pattern: str = "*.mp4",
    render: bool = False,
    skip_done: Annotated[bool, typer.Option(help="Skip clips that already have OUT/<clip>/result.json")] = False,
    max_seconds: MaxSecOpt = None,
    verbose: bool = False,
) -> None:
    """Process every clip in a folder; one failure does not stop the batch."""
    _setup_logging(verbose)
    from egovision.ingest import VideoError
    from egovision.pipeline import run_video

    cfg = _load(config, set_, max_seconds)
    log = logging.getLogger("egovision.batch")
    clips = sorted(input_dir.glob(pattern))
    summary = []
    for n, v in enumerate(clips, 1):
        if skip_done and (out / v.stem / "result.json").is_file():
            log.info("batch: [%d/%d] %s already done, skipping", n, len(clips), v.name)
            summary.append({"clip": v.stem, "ok": True, "skipped": True})
            continue
        log.info("batch: [%d/%d] %s", n, len(clips), v.name)
        try:
            r = run_video(v, out, cfg, render=render)
            summary.append({"clip": v.stem, "ok": True, **r.timings, **r.stats})
        except VideoError as e:
            log.error("batch: bad video %s: %s", v.name, e)
            summary.append({"clip": v.stem, "ok": False, "error": str(e)})
        except Exception as e:  # noqa: BLE001 - one clip's failure must never stop the batch
            log.exception("batch: failed: %s", v.name)
            summary.append({"clip": v.stem, "ok": False, "error": f"{type(e).__name__}: {e}"})
        (out / "batch_summary.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "batch_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")  # progress
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    ok = sum(s["ok"] for s in summary)
    typer.echo(f"{ok}/{len(summary)} clips succeeded; summary in {out / 'batch_summary.json'}")
    if ok < len(summary):
        raise typer.Exit(code=1)


@app.command("eval")
def eval_(
    pred: Annotated[Path, typer.Option(help="Output root with <clip>/actions.csv")] = Path("outputs"),
    labels: Annotated[Path, typer.Option(help="Folder with <clip>.actions.csv and hands_judged.csv")] = Path("labels"),
    json_out: Annotated[Path | None, typer.Option("--json")] = Path("outputs/eval_report.json"),  # noqa: UP007
) -> None:
    """Score predictions against our labels (see labels/README.md for how they were made)."""
    from egovision.evaluate import evaluate_dir, format_report

    rep = evaluate_dir(pred, labels)
    typer.echo(format_report(rep))
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(rep, indent=1), encoding="utf-8")


@app.command()
def sop(
    out: Annotated[Path, typer.Option("--out", "-o", help="Output root with <clip>/result.json")] = Path("outputs"),
    config: ConfigOpt = None,
    set_: SetOpt = [],  # noqa: B006
) -> None:
    """Stage 8: discover a step template per source and flag missing / out-of-order / repeated steps."""
    from egovision.sop import run_sop

    rep = run_sop(out, _load(config, set_, None).sop)
    for src, r in rep.items():
        typer.echo(f"{src}: {r['clips']} clips, template {[t['key'] for t in r['template']]}")
        for clip, chk in r["per_clip"].items():
            typer.echo(f"  {clip}: completeness {chk['completeness']} missing {chk['missing']} "
                       f"out_of_order {chk['out_of_order']} repeated {chk['repeated']}")
    typer.echo(f"written to {out / 'sop'}/ and <clip>/steps_check.json")


@app.command("sop-cycles")
def sop_cycles(
    out: Annotated[Path, typer.Option("--out", "-o", help="Output root with <clip>/result.json")] = Path("outputs"),
    config: ConfigOpt = None,
    set_: SetOpt = [],  # noqa: B006
) -> None:
    """Cycle consistency, labour cost and automation candidate indicator per task and step (not per person).

    Wage: --set sop_cycles.wage.currency=INR --set sop_cycles.wage.rate_per_hour=<team-declared rate>.
    """
    from egovision.sop_cycles import run_sop_cycles

    reps = run_sop_cycles(out, _load(config, set_, None).sop_cycles)
    for src, r in reps.items():
        a = r["actions"]
        typer.echo(f"{src}: {len(r['clips_used'])} clips used, {len(r['clips_excluded'])} excluded; actions used "
                   f"{a['used']}/{a['total']} (excluded {a['excluded_share']}); complete cycles {r['complete_cycles']}")
        for g in r["groups"]:
            if g["status"] != "ok":
                typer.echo(f"  {g['group']} {len(g['clips'])} clip(s): insufficient data ({g['reason']})")
                continue
            lab = g["labour"]
            cost = f"{lab['cost_per_cycle_median']} {r['wage']['currency']}" if r["wage"]["declared"] else "wage not declared"
            typer.echo(f"  {g['group']}: anchor {g['anchor']['step']}, {g['complete_cycles']} cycles, score "
                       f"{g['consistency_score']}, order distance {g['order_distance']['mean']}, cycle CV "
                       f"{g['cycle_time']['cv']}, median cycle {g['cycle_time']['median_s']} s, cost/cycle {cost}, "
                       f"task indicator: {g['indicator']}")
    typer.echo(f"written to {out / 'sop'}/<source>_consistency.json and <source>_steps.csv")


@app.command()
def trim(
    clip: Annotated[str, typer.Option("--clip", help="Clip name (results in OUT/<clip>/)")],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("outputs"),
    mode: Annotated[str | None, typer.Option(help="cut (shorter video) or keep_lite (same length); default: config")] = None,  # noqa: UP007
    force: Annotated[bool, typer.Option(help="Trim without a reviewer's approve_trim decision")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = [],  # noqa: B006
    verbose: bool = False,
) -> None:
    """Stage 10: trim + compress the sanitized copy (needs the reviewer's approval of the current plan)."""
    _setup_logging(verbose)
    from egovision.export import load_result
    from egovision.review import apply_review, load_log, trim_approved
    from egovision.trim import MODES, plan, plan_sha, run_trim

    if mode is not None and mode not in MODES:
        typer.echo(f"refused: --mode must be one of {', '.join(MODES)}")
        raise typer.Exit(code=2)
    cfg = _load(config, set_, None)
    d = out / clip
    r = load_result(d)
    log = load_log(d)
    acts = apply_review(r["actions"], log)
    sha = plan_sha(plan(acts, r["meta"].duration_s, cfg.trim))
    if cfg.trim.require_approval and not force and not trim_approved(log, sha):
        typer.echo("refused: the current trim plan has not been approved in review (or pass --force)")
        raise typer.Exit(code=2)
    if not r["sanitized"] or not Path(r["sanitized"]).is_file():
        typer.echo("refused: no sanitized copy (trimming only ever works on the sanitized video)")
        raise typer.Exit(code=2)
    info = run_trim(d, Path(r["sanitized"]), Path(r["meta"].path), acts, r["meta"].duration_s, r["meta"].fps,
                    cfg.trim, mode)  # type: ignore[arg-type]
    typer.echo(json.dumps({k: info[k] for k in ("mode", "trimmed_path", "kept_pct", "output_duration_s", "size_mb")},
                          indent=1))


@app.command()
def resanitize(
    clip: Annotated[str, typer.Option("--clip", help="Clip name, or 'all' for every clip with a privacy_log.json")],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("outputs"),
    config: ConfigOpt = None,
    set_: SetOpt = [],  # noqa: B006
    verbose: bool = False,
) -> None:
    """Stage 1: rebuild the sanitized video from privacy_log.json with the current rules (no re-detection)."""
    _setup_logging(verbose)
    from egovision.export import load_result
    from egovision.privacy import Sanitizer

    cfg = _load(config, set_, None)
    dirs = sorted(p.parent for p in out.glob("*/privacy_log.json")) if clip == "all" else [out / clip]
    for d in dirs:
        r = load_result(d)
        log = json.loads((d / "privacy_log.json").read_text(encoding="utf-8"))
        san, dropped = Sanitizer.from_log(r["meta"], cfg.privacy, log)
        dst = Path(r["sanitized"]) if r["sanitized"] else cfg.privacy.out_dir / f"{d.name}.mp4"
        summary = san.write(r["hands"], dst)
        (d / "privacy_log.json").write_text(json.dumps(san.log_record(summary), indent=1), encoding="utf-8")
        typer.echo(f"{d.name}: dropped {dropped} regions by current rules; "
                   f"blurred area {summary['mean_blurred_area_pct']}% of frame, regions {summary['regions_by_category']}")


@app.command()
def ui(host: str = "127.0.0.1", port: int = 8000, verbose: bool = False) -> None:
    """Start the dashboard (FastAPI + JS) at http://HOST:PORT. Localhost by default: footage is confidential."""
    _setup_logging(verbose)
    import uvicorn

    typer.echo(f"egovision dashboard -> http://{host}:{port}")
    uvicorn.run("egovision.api:app", host=host, port=port, log_level="info" if verbose else "warning")


@app.command()
def hands3d(video: Annotated[Path, typer.Option(help="Clip to process")],
            out: Annotated[Path, typer.Option(help="Output folder")] = Path("outputs"),
            config: ConfigOpt = None, set_: SetOpt = [],  # noqa: B006 - typer list option
            max_seconds: MaxSecOpt = None, verbose: bool = False) -> None:
    """Approximate 3D hands in camera (head) coordinates: <out>/<clip>/hands3d.jsonl + hands3d_summary.json.

    Monocular and uncalibrated (assumed fisheye FOV `hands3d.hfov_deg`); see egovision/hands3d.py.
    """
    _setup_logging(verbose)
    from egovision import hands3d as h3

    cfg = Config.load(config, set_)
    frames, summary = h3.compute(video, cfg, max_seconds=max_seconds)
    path = h3.write(frames, summary, out / video.stem)
    typer.echo(f"{path}: {summary['frames']} frames, {summary['frames_with_hand_pct']}% with a hand, "
               f"median distance {summary['median_dist_m']} m (approximate)")


@app.command()
def studio(host: str = "127.0.0.1", port: int = 8010, verbose: bool = False) -> None:
    """Start the Studio (upload a video, live Gemini analysis, worker analytics) at http://HOST:PORT.

    Uses GEMINI_API_KEY from .env; without it, the offline FORGE labeller. Settings: STUDIO_* env vars
    (egovision/studio/settings.py).
    """
    _setup_logging(verbose)
    logging.getLogger("egovision").setLevel(logging.INFO)
    import uvicorn

    typer.echo(f"egovision studio -> http://{host}:{port}")
    uvicorn.run("egovision.studio.api:create_app", factory=True, host=host, port=port,
                log_level="info" if verbose else "warning")


if __name__ == "__main__":
    app()
