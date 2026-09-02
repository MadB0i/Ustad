"""Async wrapper around the local Ollama REST API.

Ustad never runs a model itself — Ollama is the only inference engine, reached over
``http://127.0.0.1:11434``. Two behaviours here exist specifically because the teacher
and the student compete for the same GPU:

* :meth:`OllamaClient.ps` reports what is currently resident in VRAM;
* :meth:`OllamaClient.unload` evicts a model immediately via ``keep_alive: 0``, instead
  of waiting out Ollama's five-minute default.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import httpx

from . import config


class OllamaError(RuntimeError):
    """Ollama replied, but not with what we asked for."""


class OllamaUnavailable(OllamaError):
    """The daemon is not reachable at all."""


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# /api/show results, keyed by model tag. Cheap to keep and only invalidated by a re-pull.
_SHOW_CACHE: dict[str, dict[str, Any]] = {}


def strip_think(text: str) -> str:
    """Remove reasoning blocks from a teacher's answer.

    Qwen3-class models wrap chain-of-thought in ``<think>…</think>``. That is reasoning
    scaffolding, not the answer, and training a student on it teaches the wrong shape.
    An unterminated opening tag means generation was cut off mid-thought, so everything
    from that tag on is discarded too.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


@dataclass
class ChatResult:
    text: str
    done_reason: str = ""
    eval_count: int = 0
    prompt_eval_count: int = 0
    total_duration_ns: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """True when generation stopped because it hit the token cap."""
        return self.done_reason == "length"

    @property
    def tokens_per_second(self) -> float:
        seconds = self.total_duration_ns / 1e9
        return self.eval_count / seconds if seconds > 0 else 0.0


class OllamaClient:
    def __init__(self, host: str | None = None, read_timeout: float = 600.0) -> None:
        self.host = config._normalise_host(host) if host else config.OLLAMA_HOST
        self._client = httpx.AsyncClient(
            base_url=self.host,
            timeout=httpx.Timeout(connect=3.0, read=read_timeout, write=30.0, pool=5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # -- plumbing ----------------------------------------------------------------

    async def _request(self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any) -> Any:
        try:
            response = await self._client.request(
                method, path, timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT, **kwargs
            )
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaUnavailable(f"Ollama at {self.host} did not respond in time.") from exc

        if response.status_code >= 400:
            raise OllamaError(f"{method} {path} returned {response.status_code}: {response.text[:400]}")
        return response.json()

    # -- introspection -----------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Never raises — the UI needs to render a 'daemon is down' state, not a 500."""
        try:
            payload = await self._request("GET", "/api/version", timeout=2.5)
            return {"ok": True, "host": self.host, "version": payload.get("version", "")}
        except OllamaError as exc:
            return {"ok": False, "host": self.host, "error": str(exc)}

    async def list_models(self) -> list[dict[str, Any]]:
        payload = await self._request("GET", "/api/tags", timeout=15.0)
        models = payload.get("models") or []
        return sorted(models, key=lambda m: m.get("name", ""))

    async def ps(self) -> list[dict[str, Any]]:
        """Models currently loaded, with their VRAM footprint."""
        payload = await self._request("GET", "/api/ps", timeout=10.0)
        return payload.get("models") or []

    async def show(self, model: str) -> dict[str, Any]:
        """Cached ``/api/show``. Cached process-wide because the teacher list calls it
        once per model on every refresh and the answer only changes on a re-pull."""
        if model in _SHOW_CACHE:
            return _SHOW_CACHE[model]
        try:
            payload = await self._request("POST", "/api/show", json={"model": model}, timeout=20.0)
        except OllamaError:
            payload = {}
        _SHOW_CACHE[model] = payload
        return payload

    async def capabilities(self, model: str) -> list[str]:
        return list((await self.show(model)).get("capabilities") or [])

    async def has_chat_template(self, model: str) -> bool:
        """Whether the model carries a real chat template.

        Ollama reports ``completion`` for every generative model, so the capability list
        cannot distinguish an instruction-tuned model from a bare code-completion one.
        The template can: a chat model's Go template branches over ``.Messages`` or the
        individual role fields, whereas a raw completion model only interpolates
        ``.Prompt``.
        """
        template = (await self.show(model)).get("template") or ""
        return any(marker in template for marker in (".Messages", ".System", ".Role"))

    async def unload(self, model: str) -> bool:
        """Evict ``model`` from memory now. Idempotent; safe if it was never loaded."""
        try:
            await self._request(
                "POST",
                "/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
                timeout=30.0,
            )
            return True
        except OllamaError:
            return False

    async def unload_all(self) -> list[str]:
        """Free the GPU before training. Returns the models that were evicted."""
        evicted: list[str] = []
        try:
            resident = await self.ps()
        except OllamaError:
            return evicted
        for entry in resident:
            name = entry.get("model") or entry.get("name")
            if name and await self.unload(name):
                evicted.append(name)
        return evicted

    # -- generation --------------------------------------------------------------

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: dict[str, Any] | None = None,
        keep_alive: str | int = "5m",
        think: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options or {},
            "keep_alive": keep_alive,
        }
        if think is not None:
            body["think"] = think

        try:
            async with self._client.stream("POST", "/api/chat", json=body) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise OllamaError(f"/api/chat returned {response.status_code}: {detail[:400]}")
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaUnavailable(
                f"Ollama stopped responding while generating with {model}."
            ) from exc

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: dict[str, Any] | None = None,
        keep_alive: str | int = "5m",
        on_token: Callable[[str], None] | None = None,
        strip_reasoning: bool = True,
    ) -> ChatResult:
        """Run a chat completion, streaming tokens to ``on_token`` as they arrive.

        Thinking is switched off at the API level for models that advertise the
        capability, and stripped from the text regardless — older Ollama builds ignore
        the ``think`` field.
        """
        think = False if "thinking" in await self.capabilities(model) else None

        parts: list[str] = []
        final: dict[str, Any] = {}
        async for chunk in self.chat_stream(
            model, messages, options=options, keep_alive=keep_alive, think=think
        ):
            if error := chunk.get("error"):
                raise OllamaError(f"{model}: {error}")
            piece = (chunk.get("message") or {}).get("content") or ""
            if piece:
                parts.append(piece)
                if on_token is not None:
                    on_token(piece)
            if chunk.get("done"):
                final = chunk

        text = "".join(parts)
        if strip_reasoning:
            text = strip_think(text)
        return ChatResult(
            text=text.strip(),
            done_reason=final.get("done_reason", ""),
            eval_count=int(final.get("eval_count") or 0),
            prompt_eval_count=int(final.get("prompt_eval_count") or 0),
            total_duration_ns=int(final.get("total_duration") or 0),
            raw=final,
        )


def describe_model(
    entry: dict[str, Any],
    vram_total_bytes: int = 0,
    has_chat_template: bool | None = None,
) -> dict[str, Any]:
    """Flatten an /api/tags entry into what the model manager needs.

    ``fits_gpu`` is a heuristic: weights plus a working KV cache need headroom, so a
    model only counts as a fit below ~85% of total VRAM.
    """
    details = entry.get("details") or {}
    caps = list(entry.get("capabilities") or [])
    size = int(entry.get("size") or 0)
    fits = None
    if vram_total_bytes > 0 and size > 0:
        fits = size < vram_total_bytes * 0.85
    return {
        "name": entry.get("name") or entry.get("model") or "",
        "size_bytes": size,
        "parameter_size": details.get("parameter_size", ""),
        "quantization": details.get("quantization_level", ""),
        "family": details.get("family", ""),
        "context_length": details.get("context_length"),
        "capabilities": caps,
        # None when unprobed. False marks a completion-only model, which makes a
        # useless teacher — the UI greys those out rather than letting a run fail late.
        "chat_capable": has_chat_template,
        "thinking": "thinking" in caps,
        "fits_gpu": fits,
        "modified_at": entry.get("modified_at", ""),
    }
