"""FastAPI application: REST for control, SSE for telemetry, static files for the UI.

Binds to loopback with no authentication, which is correct for a single-user local tool
and wrong for anything else — there is no login because there is no network surface.

Two things happen here that are load-bearing elsewhere:

* the event bus is bound to this process's event loop at startup, which is what lets the
  training thread publish telemetry safely;
* ``config`` is imported first, so ``HF_HOME`` points into the project before
  ``transformers`` ever sees it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any

from . import config  # noqa: I001  (import order matters: sets HF_HOME)
from . import dataset as store
from . import distill
from . import export as export_module
from .events import bus
from .jobs import JobBusy, jobs
from .ollama_client import OllamaClient, OllamaError, describe_model

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

FRONTEND_DIR = config.ROOT / "frontend"

STARTED_AT = time.time()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # The training thread publishes through call_soon_threadsafe on this loop.
    bus.bind_loop(asyncio.get_running_loop())
    bus.publish("system", message="Ustad is ready.", host=f"{config.SERVER_HOST}:{config.SERVER_PORT}")
    yield
    bus.publish("system", message="shutting down")


app = FastAPI(title="Ustad", version="1.0", lifespan=lifespan)


# --------------------------------------------------------------------------------------
# Request bodies
# --------------------------------------------------------------------------------------


class DatasetRequest(BaseModel):
    name: str = "distilled"
    teacher: str
    skill: str = ""
    seeds: list[str] = Field(default_factory=list)
    target_pairs: int = 60
    system_prompt: str | None = None
    expand_temperature: float = 1.0
    answer_temperature: float = 0.7
    answer_max_tokens: int = 512
    concurrency: int = 2
    dedup_threshold: float = 0.7

    def to_config(self) -> config.DatasetConfig:
        cfg = config.DatasetConfig(
            name=self.name,
            teacher=self.teacher,
            skill=self.skill,
            seeds=[s.strip() for s in self.seeds if s.strip()],
            target_pairs=max(4, min(2000, self.target_pairs)),
            expand_temperature=self.expand_temperature,
            answer_temperature=self.answer_temperature,
            answer_max_tokens=max(64, min(4096, self.answer_max_tokens)),
            concurrency=max(1, min(8, self.concurrency)),
            dedup_threshold=self.dedup_threshold,
        )
        if self.system_prompt:
            cfg.system_prompt = self.system_prompt
        return cfg


class TrainRequest(BaseModel):
    dataset: str
    student: str = config.DEFAULT_STUDENT
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    target_modules: list[str] | None = None
    epochs: float | None = None
    max_steps: int | None = None
    batch_size: int | None = None
    grad_accum: int | None = None
    lr: float | None = None
    max_seq_len: int | None = None
    load_in_4bit: bool | None = None
    gradient_checkpointing: bool | None = None
    eval_ratio: float | None = None
    eval_every: int | None = None
    save_every: int | None = None
    seed: int | None = None

    def to_config(self) -> config.TrainConfig:
        tc = config.preset_for(config.detect_hardware()["tier"])
        tc.dataset = self.dataset
        tc.student = self.student or config.DEFAULT_STUDENT
        for key, value in self.model_dump(exclude={"dataset", "student"}).items():
            if value is not None:
                setattr(tc, key, value)
        return tc


class ExportRequest(BaseModel):
    run_id: str
    name: str = ""
    merge: bool = True


class DownloadRequest(BaseModel):
    repo_id: str


class EstimateRequest(BaseModel):
    student: str = config.DEFAULT_STUDENT
    max_seq_len: int = 512
    batch_size: int = 1
    lora_r: int = 16
    target_modules: list[str] | None = None
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True


# --------------------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------------------


@app.get("/api/system")
async def system() -> dict[str, Any]:
    hardware = config.detect_hardware()
    tier = hardware["tier"]
    async with OllamaClient(read_timeout=10.0) as client:
        health = await client.health()
        resident = []
        if health["ok"]:
            with contextlib.suppress(OllamaError):
                resident = await client.ps()

    return {
        "hardware": hardware,
        "preset": config.preset_for(tier).to_dict(),
        "tier": tier,
        "ollama": health,
        "resident": resident,
        "paths": {
            "root": str(config.ROOT),
            "data": str(config.DATA_DIR),
            "hf_cache": str(config.HF_CACHE_DIR),
            "datasets": str(config.DATASETS_DIR),
            "runs": str(config.RUNS_DIR),
            "exports": str(config.EXPORTS_DIR),
        },
        "disk_free_bytes": config.disk_free_bytes(),
        "students": [
            {**vars(model), "cached": model.repo_id in _cached_repos()} for model in config.STUDENT_CATALOG
        ],
        "default_student": config.DEFAULT_STUDENT,
        "recommended_teachers": list(config.RECOMMENDED_TEACHERS),
        "target_modules": list(config.DEFAULT_TARGET_MODULES),
        "jobs": jobs.snapshot(),
        "uptime": round(time.time() - STARTED_AT, 1),
    }


def _cached_repos() -> set[str]:
    """Which student repos are already downloaded, so the UI can say so honestly."""
    try:
        from huggingface_hub import scan_cache_dir

        return {repo.repo_id for repo in scan_cache_dir().repos if repo.repo_type == "model"}
    except Exception:
        return set()


@app.post("/api/estimate")
async def estimate(request: EstimateRequest) -> dict[str, Any]:
    tc = config.TrainConfig(
        student=request.student,
        max_seq_len=request.max_seq_len,
        batch_size=request.batch_size,
        lora_r=request.lora_r,
        load_in_4bit=request.load_in_4bit,
        gradient_checkpointing=request.gradient_checkpointing,
    )
    if request.target_modules:
        tc.target_modules = request.target_modules

    model_config = await asyncio.to_thread(_load_model_config, request.student)
    if model_config is None:
        raise HTTPException(404, f"'{request.student}' is not downloaded yet, so it cannot be measured.")
    hardware = config.detect_hardware()
    breakdown = config.estimate_peak_vram(model_config, tc)
    return {
        "student": request.student,
        "estimate": breakdown,
        "vram_total_bytes": hardware["vram_total_bytes"],
        "fits": hardware["vram_total_bytes"] == 0 or breakdown["total"] < hardware["vram_total_bytes"],
        "geometry": {
            key: model_config.get(key)
            for key in ("num_hidden_layers", "hidden_size", "vocab_size", "intermediate_size",
                        "num_attention_heads", "num_key_value_heads", "tie_word_embeddings")
        },
    }


def _load_model_config(repo_id: str) -> dict[str, Any] | None:
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(repo_id, local_files_only=True).to_dict()
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Teachers
# --------------------------------------------------------------------------------------


@app.get("/api/teachers")
async def teachers(probe: bool = True) -> dict[str, Any]:
    hardware = config.detect_hardware()
    async with OllamaClient(read_timeout=30.0) as client:
        health = await client.health()
        if not health["ok"]:
            return {"ollama": health, "models": [], "resident": [], "recommended": list(config.RECOMMENDED_TEACHERS)}
        try:
            raw = await client.list_models()
            resident = await client.ps()
        except OllamaError as exc:
            raise HTTPException(502, str(exc)) from exc

        models = []
        for entry in raw:
            name = entry.get("name") or entry.get("model") or ""
            template = await client.has_chat_template(name) if (probe and name) else None
            models.append(describe_model(entry, hardware["vram_total_bytes"], template))

    return {
        "ollama": health,
        "models": models,
        "resident": resident,
        "recommended": list(config.RECOMMENDED_TEACHERS),
    }


@app.post("/api/teachers/unload")
async def unload_teachers() -> dict[str, Any]:
    evicted = await jobs.free_gpu()
    return {"evicted": evicted}


# --------------------------------------------------------------------------------------
# Students
# --------------------------------------------------------------------------------------

_downloads: dict[str, asyncio.Task[Any]] = {}


@app.post("/api/students/download")
async def download_student(request: DownloadRequest) -> dict[str, Any]:
    repo_id = request.repo_id.strip()
    if not repo_id:
        raise HTTPException(400, "repo_id is required")
    if repo_id in _downloads and not _downloads[repo_id].done():
        return {"repo_id": repo_id, "status": "already downloading"}

    async def run() -> None:
        bus.publish("student.download", repo_id=repo_id, status="start")
        try:
            path = await asyncio.to_thread(_snapshot, repo_id)
        except Exception as exc:
            bus.publish("student.download", repo_id=repo_id, status="error", error=str(exc)[:400])
            return
        bus.publish("student.download", repo_id=repo_id, status="done", path=path)

    _downloads[repo_id] = asyncio.get_running_loop().create_task(run())
    return {"repo_id": repo_id, "status": "started"}


def _snapshot(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    # Skip the duplicate weight formats; safetensors is what transformers prefers and
    # pulling both would double a download the user is waiting on.
    return snapshot_download(
        repo_id,
        ignore_patterns=["*.bin", "*.pth", "*.msgpack", "*.h5", "*.onnx*", "*.gguf", "*.pt"],
    )


@app.post("/api/students/browse")
async def browse_model_path(request: dict[str, Any]) -> dict[str, Any]:
    """Probe a local directory or HF repo ID as a student model.
    Returns model metadata if it's a valid causal-LM, an error string otherwise."""
    path_or_id = (request.get("path") or "").strip()
    if not path_or_id:
        raise HTTPException(400, "path is required")

    def _probe() -> dict[str, Any]:
        from transformers import AutoConfig
        target = Path(path_or_id)
        local = target.exists() and target.is_dir()
        try:
            cfg = AutoConfig.from_pretrained(
                str(target) if local else path_or_id,
                local_files_only=local,
                trust_remote_code=False,
            ).to_dict()
        except Exception as exc:
            return {"valid": False, "error": str(exc)[:300]}
        arch = cfg.get("architectures", [])
        return {
            "valid": True,
            "path": str(target.resolve()) if local else path_or_id,
            "local": local,
            "hidden_size": cfg.get("hidden_size"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "vocab_size": cfg.get("vocab_size"),
            "architectures": arch,
            "model_type": cfg.get("model_type", ""),
        }

    return await asyncio.to_thread(_probe)


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


@app.get("/api/datasets")
async def datasets() -> dict[str, Any]:
    return {"datasets": store.list_datasets()}


@app.get("/api/datasets/{name}")
async def dataset_page(name: str, offset: int = 0, limit: int = 40) -> dict[str, Any]:
    try:
        return store.read_page(name, offset=offset, limit=max(1, min(200, limit)))
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no dataset named '{name}'") from exc


@app.delete("/api/datasets/{name}")
async def drop_dataset(name: str) -> dict[str, Any]:
    return {"deleted": store.delete_dataset(name)}


@app.delete("/api/datasets/{name}/rows/{row_id}")
async def drop_row(name: str, row_id: str) -> dict[str, Any]:
    deleted = store.delete_row(name, row_id)
    if not deleted:
        raise HTTPException(404, f"row '{row_id}' not found in '{name}'")
    return {"deleted": True, "summary": store.summarise(name)}


@app.post("/api/dataset/build")
async def build_dataset(request: DatasetRequest) -> dict[str, Any]:
    if not request.teacher.strip():
        raise HTTPException(400, "pick a teacher model first")
    if not request.seeds and not request.skill.strip():
        raise HTTPException(400, "give at least one seed prompt or describe the skill")
    try:
        job = jobs.start_dataset(request.to_config())
    except JobBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job": job.to_dict()}


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


@app.post("/api/train/start")
async def train_start(request: TrainRequest) -> dict[str, Any]:
    if not store.dataset_path(request.dataset).exists():
        raise HTTPException(404, f"no dataset named '{request.dataset}'")
    try:
        job = jobs.start_training(request.to_config())
    except JobBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job": job.to_dict()}


@app.post("/api/train/stop")
async def train_stop() -> dict[str, Any]:
    return {"stopping": jobs.stop()}


@app.get("/api/jobs")
async def job_state() -> dict[str, Any]:
    return jobs.snapshot()


@app.get("/api/runs")
async def runs() -> dict[str, Any]:
    return {"runs": distill.list_runs()}


@app.get("/api/runs/{run_id}")
async def run_detail(run_id: str) -> dict[str, Any]:
    try:
        return distill.read_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no run named '{run_id}'") from exc


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------


@app.get("/api/exports")
async def exports() -> dict[str, Any]:
    return {"exports": export_module.list_exports()}


@app.post("/api/export")
async def start_export(request: ExportRequest) -> dict[str, Any]:
    try:
        job = jobs.start_export(request.run_id, request.name, merge=request.merge)
    except JobBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"job": job.to_dict()}


# --------------------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------------------

_HEARTBEAT_SECONDS = 15.0


@app.get("/api/events")
async def events(request: Request, replay: int = 200) -> StreamingResponse:
    """Server-sent events. Replays recent history so a page reload catches up mid-run."""

    async def stream():
        async with bus.subscribe(replay=max(0, min(1500, replay))) as queue:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    # Comment frames keep proxies and the browser from closing an idle
                    # connection, and let the UI show a live/stale indicator.
                    yield ": heartbeat\n\n"
                    continue
                yield f"id: {event['seq']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/events/history")
async def events_history(limit: int = 300, since: int = 0) -> dict[str, Any]:
    return {"events": bus.history(limit=limit, since_seq=since), "dropped": bus.dropped}


# --------------------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.exception_handler(OllamaError)
async def ollama_error_handler(_request: Request, exc: OllamaError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    print(f"Ustad → http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"data   → {config.DATA_DIR}")
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
