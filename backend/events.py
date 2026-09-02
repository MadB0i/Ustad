"""A small publish/subscribe bus that carries telemetry to the browser over SSE.

Two callers with different threading models share this bus:

* dataset generation runs as a coroutine on the event loop;
* training runs in a worker thread, because torch blocks.

So :meth:`EventBus.publish` is safe to call from any thread. It appends to the history
under a lock and hands fan-out to the loop via ``call_soon_threadsafe`` (which is also
legal from the loop's own thread, so there is no special case).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from typing import Any, AsyncIterator, Callable, Iterable

# Enough to redraw a loss curve after a browser refresh without holding a whole run.
_HISTORY = 1500
_SUBSCRIBER_QUEUE = 2000


class EventBus:
    def __init__(self, history: int = _HISTORY) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._history: deque[dict[str, Any]] = deque(maxlen=history)
        self._sinks: list[Callable[[dict[str, Any]], None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._seq = 0
        self.dropped = 0

    # -- wiring ------------------------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the server's event loop. Until this is called (e.g. in the CLI),
        events still reach :meth:`history` and any registered sinks."""
        self._loop = loop

    def add_sink(self, sink: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Register a synchronous observer. Used by the CLI to print progress."""
        self._sinks.append(sink)
        return lambda: self._sinks.remove(sink) if sink in self._sinks else None

    # -- producing ---------------------------------------------------------------

    def publish(self, kind: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "kind": kind, "ts": time.time(), **payload}
            self._history.append(event)
            sinks = list(self._sinks)

        for sink in sinks:
            try:
                sink(event)
            except Exception:
                pass  # a broken observer must never derail a training step

        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._fan_out, event)
            except RuntimeError:
                pass  # loop shut down mid-run
        return event

    def _fan_out(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled browser tab must not grow memory without bound. Dropping is
                # safe: per-step metrics are also appended to the run's metrics.jsonl.
                self.dropped += 1

    # -- consuming ---------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def subscribe(self, replay: int = 0) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE)
        if replay > 0:
            for event in self.history(limit=replay):
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    def history(self, limit: int = 0, since_seq: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            events: Iterable[dict[str, Any]] = list(self._history)
        if since_seq:
            events = [e for e in events if e["seq"] > since_seq]
        events = list(events)
        return events[-limit:] if limit else events

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


bus = EventBus()
