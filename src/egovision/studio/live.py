"""Live updates: the analysis thread publishes, every open WebSocket of that video receives.

Messages are plain JSON dicts with a `type` (status | event | workers | analytics | batch | report | done |
error | preview). Browsers treat them as upserts, so a message that is also in the snapshot sent on connect
does no harm.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any


class LiveHub:
    def __init__(self, max_queue: int = 1000) -> None:
        self._subs: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]]] = {}
        self._lock = threading.Lock()
        self.max_queue = max_queue

    def subscribe(self, video_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(self.max_queue)
        with self._lock:
            self._subs.setdefault(video_id, set()).add((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, video_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            subs = self._subs.get(video_id, set())
            for item in [s for s in subs if s[1] is q]:
                subs.discard(item)
            if not subs:
                self._subs.pop(video_id, None)

    def subscribers(self, video_id: str) -> int:
        with self._lock:
            return len(self._subs.get(video_id, ()))

    def publish(self, video_id: str, msg: dict[str, Any]) -> None:
        """Thread-safe; never blocks the analysis thread. A subscriber that falls too far behind loses messages
        and is told to resync from the REST snapshot."""
        with self._lock:
            subs = list(self._subs.get(video_id, ()))
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(self._put, q, msg)
            except RuntimeError:  # loop closed: the socket is gone
                self.unsubscribe(video_id, q)

    @staticmethod
    def _put(q: asyncio.Queue[dict[str, Any]], msg: dict[str, Any]) -> None:
        if q.full():
            while not q.empty():
                q.get_nowait()
            q.put_nowait({"type": "resync"})
        q.put_nowait(msg)
