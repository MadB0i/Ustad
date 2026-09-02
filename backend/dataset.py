"""Turn a handful of seed prompts into a real distillation dataset.

The pipeline is response-based distillation, in three steps:

1. **Expand** — the teacher is shown the target skill plus a sample of seeds and asked
   for new, different requests in the same spirit (self-instruct).
2. **Answer** — the teacher answers every surviving prompt. Those answers *are* the
   distilled knowledge; the student never sees the teacher's weights or logits.
3. **Filter** — near-duplicates, refusals, and truncated generations are dropped.

Rows are appended to JSONL as they are produced, so a run that is cancelled halfway
still leaves a usable dataset behind.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from . import config
from .events import bus
from .ollama_client import ChatResult, OllamaClient, OllamaError

# --------------------------------------------------------------------------------------
# Near-duplicate detection
# --------------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _trigrams(tokens: list[str]) -> set[tuple[str, ...]]:
    if len(tokens) < 3:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


class Deduper:
    """Rejects prompts too similar to ones already accepted.

    Self-instruct's original filter uses ROUGE-L overlap against every prior
    instruction. Token-trigram Jaccard is a close, dependency-free stand-in: it catches
    the reworded near-copies teachers produce in bulk while leaving genuinely new
    phrasings alone.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold
        self._seen: list[set[tuple[str, ...]]] = []
        self._exact: set[str] = set()

    def __len__(self) -> int:
        return len(self._seen)

    def similarity(self, text: str) -> float:
        grams = _trigrams(_tokens(text))
        if not grams:
            return 1.0
        best = 0.0
        for other in self._seen:
            union = len(grams | other)
            if not union:
                continue
            score = len(grams & other) / union
            if score > best:
                best = score
                if best >= self.threshold:
                    break
        return best

    def accept(self, text: str) -> bool:
        """Record ``text`` and return whether it is sufficiently novel."""
        normalised = " ".join(_tokens(text))
        if not normalised or normalised in self._exact:
            return False
        if self.similarity(text) >= self.threshold:
            return False
        self._exact.add(normalised)
        self._seen.append(_trigrams(_tokens(text)))
        return True


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------

MIN_PROMPT_CHARS = 12
MAX_PROMPT_CHARS = 600
MIN_RESPONSE_CHARS = 24
MAX_RESPONSE_CHARS = 8000

# Openers that mean the teacher narrated the task instead of doing it.
_PROMPT_NOISE = (
    "here are",
    "here is a list",
    "sure,",
    "sure!",
    "certainly",
    "of course",
    "below are",
    "new requests",
    "json",
)

_REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i can not",
    "i'm unable",
    "i am unable",
    "as an ai",
    "as a language model",
    "i don't have the ability",
    "i'm not able to",
    "i apologize, but i",
)


def clean_prompt(raw: str) -> str:
    """Strip list numbering, quoting and stray markdown from a generated instruction."""
    text = raw.strip()
    text = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", text)
    text = text.strip().strip("`")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text).strip()


def prompt_is_usable(text: str) -> tuple[bool, str]:
    if len(text) < MIN_PROMPT_CHARS:
        return False, "too short"
    if len(text) > MAX_PROMPT_CHARS:
        return False, "too long"
    lowered = text.lower()
    if any(lowered.startswith(prefix) for prefix in _PROMPT_NOISE):
        return False, "meta commentary"
    if lowered.startswith(("target skill", "rules:", "example")):
        return False, "echoed the instructions"
    if not _tokens(text):
        return False, "no words"
    return True, ""


def response_is_usable(result: ChatResult, prompt: str) -> tuple[bool, str]:
    text = result.text
    if len(text) < MIN_RESPONSE_CHARS:
        return False, "too short"
    if len(text) > MAX_RESPONSE_CHARS:
        return False, "too long"
    if result.truncated:
        # A cut-off answer teaches the student to stop mid-sentence.
        return False, "hit the token cap"
    head = text[:240].lower()
    if any(marker in head for marker in _REFUSAL_MARKERS):
        return False, "refusal"
    if " ".join(_tokens(text)) == " ".join(_tokens(prompt)):
        return False, "echoed the prompt"
    return True, ""


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------

_EXPANSION_SYSTEM = (
    "You write training data. You output only what is asked for, with no preamble, "
    "commentary, or explanation."
)


def build_expansion_prompt(skill: str, examples: list[str], count: int) -> str:
    listed = "\n".join(f"{i}. {text}" for i, text in enumerate(examples, 1))
    return (
        "I am building a training set that teaches a small assistant one specific skill.\n\n"
        f"THE SKILL\n{skill.strip() or 'General helpful assistance.'}\n\n"
        f"REQUESTS ALREADY IN THE SET\n{listed}\n\n"
        f"Write {count} more requests of the same kind. Requirements:\n"
        "- Each must be genuinely different from every request listed above, and from each other.\n"
        "- Vary the phrasing, the length, the difficulty, and the sub-topic.\n"
        "- Each must stand alone, with no reference to this conversation or to the list.\n"
        "- Write the requests only. Do not answer them.\n\n"
        f"Return exactly one JSON array of {count} strings and nothing else."
    )


def parse_instructions(raw: str) -> list[str]:
    """Pull instruction strings out of a teacher reply.

    Tries JSON first because that is what was asked for, then falls back to line
    parsing, because a 3B teacher obeys the format maybe four times in five.
    """
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                items = [clean_prompt(str(x)) for x in parsed if isinstance(x, (str, int, float))]
                if items:
                    return [x for x in items if x]
        except json.JSONDecodeError:
            pass

    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        if line in "[]{}" or line.startswith("```"):
            continue
        looks_listed = bool(re.match(r"^\s*(?:\d+[\.\)]|[-*•])\s+", line))
        quoted = len(line) > 2 and line[0] == line[-1] == '"'
        if looks_listed or quoted:
            cleaned = clean_prompt(line)
            if cleaned:
                out.append(cleaned)
    return out


# --------------------------------------------------------------------------------------
# Token streaming throttle
# --------------------------------------------------------------------------------------


class _TokenThrottle:
    """Batches streamed tokens before they hit the event bus.

    A 60-pair run streams tens of thousands of tokens. Publishing each one individually
    would swamp the SSE channel with no visible benefit, so text is coalesced into
    chunks and flushed on size or elapsed time.
    """

    def __init__(self, emit: Callable[[str], None], min_chars: int = 48, min_interval: float = 0.1) -> None:
        self._emit = emit
        self._buffer: list[str] = []
        self._size = 0
        self._last = 0.0
        self.min_chars = min_chars
        self.min_interval = min_interval

    def push(self, text: str) -> None:
        self._buffer.append(text)
        self._size += len(text)
        now = time.monotonic()
        if self._size >= self.min_chars or (now - self._last) >= self.min_interval:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        payload = "".join(self._buffer)
        self._buffer.clear()
        self._size = 0
        self._last = time.monotonic()
        self._emit(payload)


# --------------------------------------------------------------------------------------
# JSONL storage
# --------------------------------------------------------------------------------------


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-._")
    return cleaned or "distilled"


def dataset_path(name: str) -> Path:
    return config.DATASETS_DIR / f"{safe_name(name)}.jsonl"


def append_row(name: str, row: dict[str, Any]) -> None:
    with dataset_path(name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_rows(name: str) -> Iterator[dict[str, Any]]:
    path = dataset_path(name)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_pairs(name: str) -> list[dict[str, str]]:
    """Training-ready pairs. Rows missing either side are skipped."""
    pairs = []
    for row in iter_rows(name):
        prompt = (row.get("prompt") or "").strip()
        response = (row.get("response") or "").strip()
        if prompt and response:
            pairs.append({"prompt": prompt, "response": response, "id": row.get("id", "")})
    return pairs


def read_page(name: str, offset: int = 0, limit: int = 50) -> dict[str, Any]:
    rows = list(iter_rows(name))
    window = rows[offset : offset + limit] if limit > 0 else rows[offset:]
    return {"name": safe_name(name), "total": len(rows), "offset": offset, "rows": window}


def delete_row(name: str, row_id: str) -> bool:
    """Drop one row. Writes to a temp file and replaces, so an interrupted delete
    cannot truncate the dataset."""
    path = dataset_path(name)
    if not path.exists():
        return False
    rows = list(iter_rows(name))
    kept = [row for row in rows if row.get("id") != row_id]
    if len(kept) == len(rows):
        return False
    temp = path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)
    return True


def delete_dataset(name: str) -> bool:
    path = dataset_path(name)
    if path.exists():
        path.unlink()
        return True
    return False


def summarise(name: str) -> dict[str, Any]:
    path = dataset_path(name)
    rows = list(iter_rows(name))
    prompt_chars = [len(r.get("prompt", "")) for r in rows]
    response_chars = [len(r.get("response", "")) for r in rows]
    teachers = sorted({r.get("teacher", "") for r in rows if r.get("teacher")})
    return {
        "name": safe_name(name),
        "rows": len(rows),
        "bytes": path.stat().st_size if path.exists() else 0,
        "modified": path.stat().st_mtime if path.exists() else 0,
        "teachers": teachers,
        "avg_prompt_chars": round(sum(prompt_chars) / len(prompt_chars), 1) if prompt_chars else 0,
        "avg_response_chars": round(sum(response_chars) / len(response_chars), 1) if response_chars else 0,
    }


def list_datasets() -> list[dict[str, Any]]:
    names = sorted(p.stem for p in config.DATASETS_DIR.glob("*.jsonl"))
    return [summarise(n) for n in names]


# --------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------


@dataclass
class BuildStats:
    requested: int = 0
    prompts_generated: int = 0
    prompts_kept: int = 0
    pairs_written: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    expansion_rounds: int = 0
    teacher_tokens: int = 0
    started_at: float = field(default_factory=time.time)
    cancelled: bool = False

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "prompts_generated": self.prompts_generated,
            "prompts_kept": self.prompts_kept,
            "pairs_written": self.pairs_written,
            "rejected": dict(sorted(self.rejected.items(), key=lambda kv: -kv[1])),
            "expansion_rounds": self.expansion_rounds,
            "teacher_tokens": self.teacher_tokens,
            "elapsed": round(time.time() - self.started_at, 1),
            "cancelled": self.cancelled,
        }


MAX_EXPANSION_ROUNDS = 12
_FEWSHOT_SEEDS = 6
_FEWSHOT_GENERATED = 4


async def expand_seeds(
    client: OllamaClient,
    cfg: config.DatasetConfig,
    stats: BuildStats,
    deduper: Deduper,
) -> list[str]:
    """Grow the seed prompts into ``cfg.target_pairs`` distinct instructions."""
    rng = random.Random(cfg.seed or None)
    accepted: list[str] = []

    # The user's own seeds are real examples of the target skill, so they belong in the
    # training set rather than being used only as few-shot fodder.
    for seed in cfg.seeds:
        cleaned = clean_prompt(seed)
        ok, reason = prompt_is_usable(cleaned)
        if not ok:
            stats.reject(f"seed: {reason}")
            continue
        if deduper.accept(cleaned):
            accepted.append(cleaned)

    bus.publish(
        "dataset.phase",
        phase="expand",
        message=f"{len(accepted)} seed prompts accepted; expanding to {cfg.target_pairs}",
        accepted=len(accepted),
        target=cfg.target_pairs,
    )

    rounds = 0
    stall = 0
    while len(accepted) < cfg.target_pairs and rounds < MAX_EXPANSION_ROUNDS:
        rounds += 1
        stats.expansion_rounds = rounds
        remaining = cfg.target_pairs - len(accepted)
        ask = max(6, min(20, remaining + 4))

        examples = list(cfg.seeds[:_FEWSHOT_SEEDS])
        pool = accepted[len(cfg.seeds) :]
        if pool:
            examples += rng.sample(pool, k=min(_FEWSHOT_GENERATED, len(pool)))
        if not examples:
            examples = accepted[:_FEWSHOT_SEEDS] or ["Explain the topic clearly."]

        bus.publish(
            "dataset.phase",
            phase="expand",
            message=f"round {rounds}: asking {cfg.teacher} for {ask} new prompts",
            accepted=len(accepted),
            target=cfg.target_pairs,
            round=rounds,
        )

        result = await client.chat(
            cfg.teacher,
            [
                {"role": "system", "content": _EXPANSION_SYSTEM},
                {"role": "user", "content": build_expansion_prompt(cfg.skill, examples, ask)},
            ],
            options={
                "temperature": cfg.expand_temperature,
                "top_p": 0.95,
                "num_predict": 1200,
                **({"seed": cfg.seed + rounds} if cfg.seed else {}),
            },
        )
        stats.teacher_tokens += result.eval_count

        candidates = parse_instructions(result.text)
        stats.prompts_generated += len(candidates)
        gained = 0
        for candidate in candidates:
            if len(accepted) >= cfg.target_pairs:
                break
            ok, reason = prompt_is_usable(candidate)
            if not ok:
                stats.reject(reason)
                continue
            if not deduper.accept(candidate):
                stats.reject("near-duplicate")
                continue
            accepted.append(candidate)
            gained += 1
            bus.publish("dataset.prompt", index=len(accepted), total=cfg.target_pairs, prompt=candidate)

        if gained == 0:
            stall += 1
            # Two barren rounds in a row means the teacher has run out of angles on this
            # skill; more rounds would just burn time producing duplicates.
            if stall >= 2:
                bus.publish(
                    "dataset.phase",
                    phase="expand",
                    message="teacher stopped producing novel prompts; continuing with what we have",
                    accepted=len(accepted),
                    target=cfg.target_pairs,
                )
                break
        else:
            stall = 0

    stats.prompts_kept = len(accepted)
    return accepted


async def answer_prompts(
    client: OllamaClient,
    cfg: config.DatasetConfig,
    prompts: list[str],
    stats: BuildStats,
) -> None:
    """Have the teacher answer every prompt, writing surviving pairs to JSONL."""
    semaphore = asyncio.Semaphore(max(1, cfg.concurrency))
    write_lock = asyncio.Lock()
    counter = {"done": 0}
    total = len(prompts)

    async def one(index: int, prompt: str) -> None:
        async with semaphore:
            slot = f"s{index % max(1, cfg.concurrency)}"
            bus.publish("dataset.generating", index=index, total=total, prompt=prompt, slot=slot)
            throttle = _TokenThrottle(
                lambda text: bus.publish("dataset.token", index=index, slot=slot, text=text)
            )
            try:
                result = await client.chat(
                    cfg.teacher,
                    [
                        {"role": "system", "content": cfg.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    options={
                        "temperature": cfg.answer_temperature,
                        "top_p": 0.9,
                        "num_predict": cfg.answer_max_tokens,
                        **({"seed": cfg.seed + index} if cfg.seed else {}),
                    },
                    on_token=throttle.push,
                )
            except OllamaError as exc:
                stats.reject("teacher error")
                bus.publish("dataset.rejected", index=index, total=total, reason=str(exc)[:200], prompt=prompt)
                return
            finally:
                throttle.flush()

            stats.teacher_tokens += result.eval_count
            ok, reason = response_is_usable(result, prompt)
            if not ok:
                stats.reject(reason)
                bus.publish("dataset.rejected", index=index, total=total, reason=reason, prompt=prompt)
                return

            row = {
                "id": uuid.uuid4().hex[:12],
                "prompt": prompt,
                "response": result.text,
                "teacher": cfg.teacher,
                "source": "seed" if index < len(cfg.seeds) else "self-instruct",
                "created_at": time.time(),
                "teacher_tokens": result.eval_count,
            }
            async with write_lock:
                append_row(cfg.name, row)
                stats.pairs_written += 1
                counter["done"] += 1
            bus.publish(
                "dataset.pair",
                index=counter["done"],
                total=total,
                row=row,
                tokens_per_second=round(result.tokens_per_second, 1),
            )

    bus.publish("dataset.phase", phase="answer", message=f"generating {total} responses", total=total)
    await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))


async def build_dataset(cfg: config.DatasetConfig, client: OllamaClient | None = None) -> dict[str, Any]:
    """Run the full seed → expand → answer → filter pipeline."""
    stats = BuildStats(requested=cfg.target_pairs)
    owns_client = client is None
    client = client or OllamaClient()
    cfg.name = safe_name(cfg.name)

    bus.publish(
        "dataset.phase",
        phase="start",
        message=f"building '{cfg.name}' with teacher {cfg.teacher}",
        dataset=cfg.name,
        teacher=cfg.teacher,
        target=cfg.target_pairs,
    )
    try:
        deduper = Deduper(cfg.dedup_threshold)
        # Prompts already in the file must not be regenerated when topping up a dataset.
        existing = 0
        for row in iter_rows(cfg.name):
            if prompt := row.get("prompt"):
                deduper.accept(prompt)
                existing += 1
        if existing:
            bus.publish(
                "dataset.phase",
                phase="expand",
                message=f"'{cfg.name}' already has {existing} rows; new prompts will avoid them",
                accepted=0,
                target=cfg.target_pairs,
            )

        prompts = await expand_seeds(client, cfg, stats, deduper)
        if not prompts:
            raise OllamaError(
                "No usable prompts survived expansion. Check that the teacher is an "
                "instruction-tuned chat model and that the seeds are real requests."
            )
        await answer_prompts(client, cfg, prompts, stats)
    except asyncio.CancelledError:
        stats.cancelled = True
        bus.publish("dataset.done", dataset=cfg.name, stats=stats.to_dict(), cancelled=True)
        raise
    except Exception as exc:
        bus.publish("dataset.error", dataset=cfg.name, error=str(exc)[:500], stats=stats.to_dict())
        raise
    finally:
        if owns_client:
            await client.aclose()

    summary = summarise(cfg.name)
    bus.publish("dataset.done", dataset=cfg.name, stats=stats.to_dict(), summary=summary)
    return {"dataset": cfg.name, "stats": stats.to_dict(), "summary": summary, "path": str(dataset_path(cfg.name))}
