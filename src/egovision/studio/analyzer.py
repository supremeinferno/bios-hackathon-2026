"""Provider-neutral part of the batch analysis: the instructions, the per-batch context, the result type and
the errors. A provider (egovision/studio/gemini.py) turns `parts(ctx)` into its own request format; the
backend then validates the JSON reply with Pydantic (models.py) exactly the same way for any provider.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from egovision.studio.frames import SampledFrame
from egovision.studio.models import EventRecord
from egovision.studio.settings import ACTIONS

SYSTEM_PROMPT = """You analyse frames sampled from industrial and factory video and report what each worker is doing, \
as a timeline of tasks. Your reply is stored in a database and used to measure task durations, so timestamps and \
consistency matter more than prose.

Frames
- Each image is preceded by a label "Frame k · t=SS.S s". Use those timestamps exactly: start_ts is the timestamp of \
the first frame in which the task is visible, end_ts the timestamp of the last frame in which it is visible (equal to \
start_ts if it is visible in one frame only).
- A "context frame" (if present) was already analysed in the previous batch; use it only to understand continuity. \
Never report an event that is visible only in the context frame.

Workers
- Worker IDs are W1, W2, W3, ... Reuse the IDs from "Known workers" for the same people.
- In head-mounted (egocentric) footage the camera wearer is W1 with is_camera_wearer=true: their hands and arms \
appear at the bottom of the image and they are never fully visible. Other people get the next free ID.
- In `workers` list only workers that are new in this batch or whose description you can improve. Describe a worker \
only by clothing, protective equipment, position and role (e.g. "camera wearer, black gloves, at assembly bench"). \
Never describe faces, skin, age or other identifying traits, and never guess names.

Events
- One event = one worker doing one action on one object without a break. Start a new event when the action or the \
object changes. Different workers may have overlapping events; one worker should normally have one event at a time.
- action: exactly one of the fixed vocabulary below. Report idle time as "wait" (visible for at least 2 frames) and \
moving between positions as "walk".
- object: the workpiece, part or material handled, as a short lower-case singular noun phrase (1-3 words). Keep \
names consistent with "Open events" and "Recent events" when it is the same thing.
- tool: the hand tool, power tool or machine used (e.g. "screwdriver", "drill", "scanner"), or null for bare hands.
- status: "completed" if the task visibly finishes within these frames; "in_progress" if it is still going on in \
the last frame of this batch; "interrupted" if it stops before it is finished (abandoned, put down unfinished).
- continues_event_id: if the worker is still doing one of the "Open events" at the start of this batch, report it \
again with that event's id, and with start_ts/end_ts covering only the frames of this batch. Otherwise null.
- confidence (0-1): 0.9+ clearly visible, about 0.7 partly occluded but clear from context, 0.5 or less a guess.
- sop_step: when an SOP is given, the 1-based number of the SOP step this event carries out, or null if none \
matches. Without an SOP always null.
- evidence: one short sentence on what is visible that supports the event.
- Report only what is visible. If nothing happens, return an empty events list. Never invent tasks to fill time.

Action vocabulary
""" + "\n".join(f"- {k}: {v}" for k, v in ACTIONS.items())


class AnalyzerNotConfigured(RuntimeError):
    """No API key: the run goes to the offline labeller instead."""


class AnalyzerFatal(RuntimeError):
    """An error every later batch would hit too (invalid key, unknown model)."""


class BatchFailed(RuntimeError):
    """This batch produced no usable reply (blocked, truncated, API error after retries and fallbacks)."""

    def __init__(self, msg: str, raw: dict[str, Any] | None = None) -> None:
        super().__init__(msg)
        self.raw = raw


@dataclass
class BatchContext:
    video_label: str
    duration_s: float
    sample_every_s: float
    batch_index: int
    n_batches: int
    frames: list[SampledFrame]
    context_frame: SampledFrame | None = None
    workers: list[dict[str, Any]] = field(default_factory=list)
    open_events: list[EventRecord] = field(default_factory=list)
    recent_events: list[EventRecord] = field(default_factory=list)
    sop: list[str] | None = None


@dataclass
class BatchResult:
    reply: dict[str, Any]
    raw: dict[str, Any]  # full API response, saved to disk
    model: str  # model name that answered
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class Analyzer(Protocol):
    model: str

    def analyze(self, ctx: BatchContext) -> BatchResult: ...


def _fmt_event(e: EventRecord) -> str:
    tool = f", tool {e.tool}" if e.tool else ""
    return (f"- {e.event_id} · {e.worker_id} · {e.action} · {e.object}{tool} · {e.start_ts:.1f}-{e.end_ts:.1f} s"
            f" ({e.completion_status})")


def batch_header(ctx: BatchContext) -> str:
    t0, t1 = ctx.frames[0].t, ctx.frames[-1].t
    lines = [f'Video "{ctx.video_label}", {ctx.duration_s:.1f} s long, one frame every {ctx.sample_every_s:g} s.',
             f"Batch {ctx.batch_index + 1} of {ctx.n_batches}: t={t0:.1f}-{t1:.1f} s ({len(ctx.frames)} frames).", ""]
    lines.append("Known workers:")
    lines += [f"- {w['worker_id']}: {w['description']}" for w in ctx.workers] or ["- none yet"]
    lines += ["", "Open events (still in progress at the end of the previous batch):"]
    lines += [f"- {e.event_id} · {e.worker_id} · {e.action} · {e.object}" + (f" · tool {e.tool}" if e.tool else "")
              + f" · started {e.start_ts:.1f} s" for e in ctx.open_events] or ["- none"]
    lines += ["", "Recent events (for consistent naming):"]
    lines += [_fmt_event(e) for e in ctx.recent_events] or ["- none"]
    lines.append("")
    if ctx.sop:
        lines.append("SOP (standard operating procedure) steps:")
        lines += [f"{i}. {s}" for i, s in enumerate(ctx.sop, 1)]
    else:
        lines.append("No SOP was provided: sop_step must be null.")
    return "\n".join(lines)


def parts(ctx: BatchContext) -> Iterator[tuple[Literal["text", "image"], Any]]:
    """The request body in order: header, optional context frame, then each frame with its timestamp label."""
    yield "text", batch_header(ctx)
    if ctx.context_frame is not None:
        yield "text", f"Context frame · t={ctx.context_frame.t:.1f} s (already analysed)"
        yield "image", ctx.context_frame.jpeg
    for k, f in enumerate(ctx.frames, 1):
        yield "text", f"Frame {k} · t={f.t:.1f} s"
        yield "image", f.jpeg
    yield "text", "Report the workers and events for this batch."
