"""One job at a time.

Dataset generation needs the teacher resident in VRAM; training needs the student
resident in VRAM. On a 4 GB card they do not both fit, and Ollama holds a model for five
minutes after its last request by default. So the two phases are serialised through a
single slot, and the teacher is explicitly evicted before training starts.

This also gives the UI something honest to render: exactly one job, its phase, and
whether it can be cancelled.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import config
from . import dataset as store
from . import distill
from .events import bus
from .ollama_client import OllamaClient, OllamaError


class JobBusy(RuntimeError):
    """Something else already holds the slot."""


@dataclass
class JobState:
    id: str
    kind: str  # "dataset" | "train" | "export"
    label: str
    status: str = "running"  # running | done | cancelled | error
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": round((self.finished_at or time.time()) - self.started_at, 1),
            "error": self.error,
            "result": self.result,
        }


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current: JobState | None = None
        self.last: JobState | None = None
        self._task: asyncio.Task[Any] | None = None
        self._run: distill.TrainingRun | None = None
        self._counter = 0

    # -- state -------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self.current is not None

    def snapshot(self) -> dict[str, Any]:
        current = self.current
        return {
            "busy": current is not None,
            "current": current.to_dict() if current else None,
            "last": self.last.to_dict() if self.last else None,
            "cancellable": bool(current),
        }

    def _publish_state(self) -> None:
        bus.publish("job.state", **self.snapshot())

    def _claim(self, kind: str, label: str) -> JobState:
        with self._lock:
            if self.current is not None:
                raise JobBusy(
                    f"'{self.current.label}' is still running. Stop it before starting another job."
                )
            self._counter += 1
            self.current = JobState(id=f"job-{self._counter}", kind=kind, label=label)
            return self.current

    def _release(self, job: JobState, status: str, *, error: str | None = None, result: Any = None) -> None:
        with self._lock:
            job.status = status
            job.finished_at = time.time()
            job.error = error
            job.result = result if isinstance(result, dict) else None
            self.last = job
            self.current = None
            self._task = None
            self._run = None
        self._publish_state()

    # -- launching ---------------------------------------------------------------

    async def _supervise(self, job: JobState, coro: Awaitable[Any]) -> Any:
        try:
            result = await coro
        except asyncio.CancelledError:
            self._release(job, "cancelled")
            raise
        except Exception as exc:
            bus.publish("job.error", id=job.id, job_kind=job.kind, error=str(exc)[:600])
            self._release(job, "error", error=str(exc)[:600])
            traceback.print_exc()
            return None
        self._release(job, "done", result=result)
        return result

    def start_dataset(self, cfg: config.DatasetConfig) -> JobState:
        job = self._claim("dataset", f"building dataset '{cfg.name}'")
        self._publish_state()
        self._task = asyncio.get_running_loop().create_task(
            self._supervise(job, store.build_dataset(cfg))
        )
        return job

    def start_training(self, tc: config.TrainConfig) -> JobState:
        job = self._claim("train", f"training on '{tc.dataset}'")
        self._publish_state()
        self._task = asyncio.get_running_loop().create_task(self._supervise(job, self._train(tc)))
        return job

    def start_export(self, run_id: str, name: str, *, merge: bool = True) -> JobState:
        from . import export as export_module

        job = self._claim("export", f"exporting '{run_id}'")
        self._publish_state()
        self._task = asyncio.get_running_loop().create_task(
            self._supervise(
                job,
                asyncio.to_thread(export_module.export_run, run_id, name, merge),
            )
        )
        return job

    # -- training -----------------------------------------------------------------

    async def _train(self, tc: config.TrainConfig) -> dict[str, Any]:
        """Free the GPU of any resident teacher, then train in a worker thread."""
        await self.free_gpu()

        run = distill.TrainingRun(tc)
        self._run = run
        try:
            return await asyncio.to_thread(run.run)
        except asyncio.CancelledError:
            run.stop()
            raise

    async def free_gpu(self) -> list[str]:
        """Evict every model Ollama is holding. Reversible: the next request reloads it."""
        try:
            async with OllamaClient(read_timeout=60.0) as client:
                evicted = await client.unload_all()
        except OllamaError:
            return []
        if evicted:
            bus.publish(
                "system",
                message=f"Freed the GPU by unloading {', '.join(evicted)}.",
                evicted=evicted,
            )
            # Ollama's unload returns before the driver has actually released the
            # allocation; a short pause keeps the first CUDA alloc from racing it.
            await asyncio.sleep(1.5)
        return evicted

    # -- stopping -----------------------------------------------------------------

    def stop(self) -> bool:
        """Ask the running job to stop. Training stops cooperatively at the next
        micro-batch so the adapter is still saved; other jobs are cancelled outright."""
        job = self.current
        if job is None:
            return False
        if self._run is not None:
            self._run.stop()
            bus.publish("job.stopping", id=job.id, job_kind=job.kind)
            return True
        if self._task is not None:
            self._task.cancel()
            return True
        return False


jobs = JobManager()
