"""Command line: `egovision run | batch | eval | ui`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from egovision.config import Config

app = typer.Typer(add_completion=False, help="Egocentric factory video -> hand tracks, contacts and a task-step timeline.")

ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="YAML config file.")]  # noqa: UP007
SetOpt = Annotated[list[str], typer.Option("--set", "-s", help="Override, e.g. --set ingest.stride=3")]
MaxSecOpt = Annotated[Optional[float], typer.Option(help="Only process the first N seconds")]  # noqa: UP007


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
        except (VideoError, RuntimeError, OSError, ValueError) as e:
            log.error("batch: failed: %s: %s", v.name, e)
            summary.append({"clip": v.stem, "ok": False, "error": str(e)})
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
    json_out: Annotated[Optional[Path], typer.Option("--json")] = Path("outputs/eval_report.json"),  # noqa: UP007
) -> None:
    """Score predictions against our labels (see labels/README.md for how they were made)."""
    from egovision.evaluate import evaluate_dir, format_report

    rep = evaluate_dir(pred, labels)
    typer.echo(format_report(rep))
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(rep, indent=1), encoding="utf-8")


@app.command()
def ui(host: str = "127.0.0.1", port: int = 8000, verbose: bool = False) -> None:
    """Start the dashboard (FastAPI + JS) at http://HOST:PORT. Localhost by default: footage is confidential."""
    _setup_logging(verbose)
    import uvicorn

    typer.echo(f"egovision dashboard -> http://{host}:{port}")
    uvicorn.run("egovision.api:app", host=host, port=port, log_level="info" if verbose else "warning")


if __name__ == "__main__":
    app()
