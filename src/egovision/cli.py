"""Command line: `egovision run`, `egovision batch`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from egovision.config import Config

app = typer.Typer(add_completion=False, help="Egocentric factory video -> hand tracks + task-step timeline.")

ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="YAML config file.")]  # noqa: UP007
SetOpt = Annotated[list[str], typer.Option("--set", "-s", help="Override, e.g. --set ingest.stride=3")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


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
    max_seconds: Annotated[Optional[float], typer.Option(help="Only process the first N seconds")] = None,  # noqa: UP007
    verbose: bool = False,
) -> None:
    """Process one clip."""
    _setup_logging(verbose)
    from egovision.pipeline import run_video

    res = run_video(video, out, _load(config, set_, max_seconds), render=render)
    typer.echo(json.dumps({"out_dir": str(res.out_dir), "timings": res.timings, "stats": res.stats}, indent=1))


@app.command()
def batch(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("outputs"),
    config: ConfigOpt = None,
    set_: SetOpt = [],  # noqa: B006
    pattern: str = "*.mp4",
    render: bool = False,
    max_seconds: Annotated[Optional[float], typer.Option()] = None,  # noqa: UP007
    verbose: bool = False,
) -> None:
    """Process every clip in a folder; one failure does not stop the batch."""
    _setup_logging(verbose)
    from egovision.ingest import VideoError
    from egovision.pipeline import run_video

    cfg = _load(config, set_, max_seconds)
    summary = []
    for v in sorted(input_dir.glob(pattern)):
        try:
            r = run_video(v, out, cfg, render=render)
            summary.append({"clip": v.stem, "ok": True, **r.timings, **r.stats})
        except (VideoError, RuntimeError, OSError, ValueError) as e:
            logging.getLogger("egovision.batch").error("failed: %s: %s", v.name, e)
            summary.append({"clip": v.stem, "ok": False, "error": str(e)})
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    ok = sum(s["ok"] for s in summary)
    typer.echo(f"{ok}/{len(summary)} clips succeeded; summary in {out / 'batch_summary.json'}")
    if ok < len(summary):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
