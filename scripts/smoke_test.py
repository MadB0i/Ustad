"""Command-line verification for Ustad. Everything here touches the real machine.

    python scripts/smoke_test.py --check                     hardware + Ollama preflight
    python scripts/smoke_test.py --build --n 24              generate a real dataset
    python scripts/smoke_test.py --train --max-steps 30      real QLoRA training
    python scripts/smoke_test.py --export                    merge and generate from it
    python scripts/smoke_test.py --all                       all four, in order

No mocks, no fixtures: if a step prints a number, a GPU or a teacher produced it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402  (must precede any transformers import)
from backend import dataset as store  # noqa: E402
from backend.events import bus  # noqa: E402
from backend.ollama_client import OllamaClient, OllamaError, describe_model  # noqa: E402

DATASET_NAME = "smoke"

SKILL = (
    "Explain Python error messages: state what the error means, the most likely cause, "
    "and the concrete fix, in a short paragraph followed by a corrected code snippet."
)

SEEDS = [
    "What does \"TypeError: unhashable type: 'list'\" mean and how do I fix it?",
    "Explain \"IndentationError: unexpected indent\" and how to resolve it.",
    "Why do I get \"ValueError: too many values to unpack (expected 2)\" and what is the fix?",
    "What causes \"AttributeError: 'NoneType' object has no attribute 'append'\"?",
    "Explain \"ModuleNotFoundError: No module named 'requests'\" and how to fix it.",
    "Why does \"RecursionError: maximum recursion depth exceeded\" happen?",
]

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def gb(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


def rule(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}\n{'-' * max(24, len(title))}")


def ok(passed: bool) -> str:
    return f"{GREEN}pass{RESET}" if passed else f"{RED}FAIL{RESET}"


# --------------------------------------------------------------------------------------
# 1. preflight
# --------------------------------------------------------------------------------------


def check() -> bool:
    rule("environment")
    print(f"python          {sys.version.split()[0]}")
    print(f"HF_HOME         {os.environ.get('HF_HOME')}")
    print(f"data dir        {config.DATA_DIR}")
    print(f"disk free       {gb(config.disk_free_bytes())}")

    hardware = config.detect_hardware()
    rule("torch / cuda")
    print(f"torch           {hardware['torch']}  (cuda build {hardware['cuda_build']})")
    print(f"cuda available  {hardware['cuda_available']}")
    print(f"device          {hardware['device_name']}  capability {hardware['capability']}")
    print(f"arch list       {', '.join(hardware['arch_list']) or '(none)'}")
    print(f"arch supported  {hardware['arch_supported']}")
    print(f"vram            {gb(hardware['vram_total_bytes'])} total, {gb(hardware['vram_free_bytes'])} free")
    print(f"bf16            {hardware['supports_bf16']}")
    print(f"bitsandbytes    {hardware['bitsandbytes'] or hardware['bitsandbytes_error']}")
    print(f"tier            {BOLD}{hardware['tier']}{RESET}")
    for note in hardware["notes"]:
        print(f"{DIM}  note: {note}{RESET}")

    passed = True

    # arch_list is necessary but not sufficient: the only proof a kernel exists is
    # running one. Both matmuls below are the shapes training will actually use.
    if hardware["cuda_available"]:
        rule("live gpu kernel test")
        import torch

        try:
            a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
            value = float((a @ a).sum())
            print(f"fp16 matmul     {ok(True)}  (sum={value:.1f})")
        except Exception as exc:
            print(f"fp16 matmul     {ok(False)}  {exc}")
            passed = False

        try:
            import bitsandbytes as bnb

            layer = bnb.nn.Linear4bit(
                256, 256, bias=False, compute_dtype=torch.float16, quant_type="nf4"
            ).to("cuda")
            x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
            out = layer(x)
            print(f"nf4 Linear4bit  {ok(True)}  (out {tuple(out.shape)} {out.dtype})")
        except Exception as exc:
            print(f"nf4 Linear4bit  {ok(False)}  {exc}")
            passed = False
    else:
        print(f"{DIM}No CUDA device; the CPU preset will be used.{RESET}")

    preset = config.preset_for(hardware["tier"])
    rule(f"preset: {hardware['tier']}")
    print(
        f"lora r={preset.lora_r} alpha={preset.lora_alpha}  seq={preset.max_seq_len}  "
        f"batch={preset.batch_size}x{preset.grad_accum}  lr={preset.lr}  "
        f"4bit={preset.load_in_4bit}  checkpointing={preset.gradient_checkpointing}"
    )

    rule("ollama")
    health, models = asyncio.run(_ollama_probe(hardware["vram_total_bytes"]))
    print(f"daemon          {health.get('host')}  {ok(health.get('ok', False))}  {health.get('version') or health.get('error', '')}")
    if models:
        for entry in models:
            fit = {True: "fits", False: "too big", None: "?"}[entry["fits_gpu"]]
            chat = {True: "chat", False: "completion-only", None: "?"}[entry["chat_capable"]]
            print(
                f"  {entry['name']:<24} {gb(entry['size_bytes']):>9}  {entry['parameter_size'] or '?':>6}  "
                f"{fit:<8} {chat}"
            )
    elif health.get("ok"):
        print(f"{DIM}  No models pulled. Try: ollama pull qwen2.5:3b{RESET}")
        passed = False
    else:
        passed = False

    if hardware["arch_supported"] is False:
        passed = False

    rule("result")
    print(f"preflight       {ok(passed)}")
    return passed


async def _ollama_probe(vram_total: int) -> tuple[dict, list[dict]]:
    async with OllamaClient(read_timeout=30.0) as client:
        health = await client.health()
        if not health["ok"]:
            return health, []
        try:
            raw = await client.list_models()
        except OllamaError as exc:
            return {"ok": False, "host": client.host, "error": str(exc)}, []
        described = []
        for entry in raw:
            name = entry.get("name") or entry.get("model") or ""
            template = await client.has_chat_template(name) if name else None
            described.append(describe_model(entry, vram_total, template))
        return health, described


# --------------------------------------------------------------------------------------
# 2. dataset
# --------------------------------------------------------------------------------------


def build(teacher: str, n: int, name: str) -> bool:
    from backend.config import DatasetConfig

    if not teacher:
        teacher = asyncio.run(_pick_teacher())
        if not teacher:
            print(f"{RED}No chat-capable teacher available. Run: ollama pull qwen2.5:3b{RESET}")
            return False

    rule(f"building '{name}' with {teacher}, target {n} pairs")
    unregister = bus.add_sink(_print_dataset_event)
    started = time.time()
    try:
        result = asyncio.run(
            store.build_dataset(
                DatasetConfig(name=name, teacher=teacher, skill=SKILL, seeds=list(SEEDS), target_pairs=n)
            )
        )
    except (OllamaError, RuntimeError) as exc:
        print(f"\n{RED}build failed: {exc}{RESET}")
        return False
    finally:
        unregister()

    stats, summary = result["stats"], result["summary"]
    rule("dataset")
    print(f"path            {result['path']}")
    print(f"rows            {summary['rows']}  ({gb(summary['bytes']) if summary['bytes'] > 1e6 else str(summary['bytes']) + ' bytes'})")
    print(f"prompts         generated {stats['prompts_generated']}, kept {stats['prompts_kept']}")
    print(f"pairs written   {stats['pairs_written']} in {stats['expansion_rounds']} expansion rounds")
    print(f"rejected        {json.dumps(stats['rejected']) if stats['rejected'] else 'nothing'}")
    print(f"avg prompt      {summary['avg_prompt_chars']} chars")
    print(f"avg response    {summary['avg_response_chars']} chars")
    print(f"teacher tokens  {stats['teacher_tokens']:,}")
    print(f"elapsed         {time.time() - started:.1f}s")

    rows = list(store.iter_rows(name))
    for row in rows[:3]:
        rule(f"sample {row['id'][:8]}")
        print(f"{BOLD}Q{RESET} {row['prompt']}")
        body = row["response"]
        print(f"{BOLD}A{RESET} {body[:700]}{'…' if len(body) > 700 else ''}")

    passed = summary["rows"] >= max(1, int(n * 0.8))
    rule("result")
    print(f"dataset build   {ok(passed)}  ({summary['rows']}/{n} rows)")
    return passed


async def _pick_teacher() -> str:
    async with OllamaClient(read_timeout=30.0) as client:
        if not (await client.health())["ok"]:
            return ""
        for entry in await client.list_models():
            name = entry.get("name") or ""
            if name and await client.has_chat_template(name):
                return name
    return ""


def _print_dataset_event(event: dict) -> None:
    kind = event["kind"]
    if kind == "dataset.phase":
        print(f"\n{DIM}[{event.get('phase')}]{RESET} {event.get('message', '')}")
    elif kind == "dataset.prompt":
        print(f"  + {event['prompt'][:96]}")
    elif kind == "dataset.generating":
        print(f"{DIM}  answering: {event['prompt'][:80]}{RESET}")
    elif kind == "dataset.pair":
        row = event["row"]
        print(
            f"  {GREEN}✓{RESET} pair {event['index']}/{event['total']}  "
            f"{len(row['response'])} chars, {row['teacher_tokens']} tok, "
            f"{event['tokens_per_second']} tok/s"
        )
    elif kind == "dataset.rejected":
        print(f"  {RED}✗{RESET} {event.get('reason')}: {str(event.get('prompt', ''))[:70]}")
    elif kind == "dataset.error":
        print(f"{RED}error: {event.get('error')}{RESET}")


# --------------------------------------------------------------------------------------
# 3. training
# --------------------------------------------------------------------------------------


def train(name: str, student: str, max_steps: int, seq: int) -> bool:
    from backend import distill

    hardware = config.detect_hardware()
    tc = config.preset_for(hardware["tier"])
    tc.dataset = name
    tc.student = student or config.DEFAULT_STUDENT
    tc.max_steps = max_steps
    if seq:
        tc.max_seq_len = seq
    tc.eval_every = max(5, max_steps // 3) if max_steps else tc.eval_every
    tc.save_every = 0

    rule(f"training {tc.student} on '{name}'")
    print(f"{DIM}evicting any resident Ollama model to free VRAM…{RESET}")
    evicted = asyncio.run(_free_gpu())
    print(f"{DIM}evicted: {', '.join(evicted) or 'nothing was loaded'}{RESET}")

    unregister = bus.add_sink(_print_train_event)
    run = distill.TrainingRun(tc)
    try:
        result = run.run()
    except Exception as exc:
        print(f"\n{RED}training failed: {type(exc).__name__}: {exc}{RESET}")
        return False
    finally:
        unregister()

    rule("run summary")
    for key in (
        "run_id", "steps_completed", "micro_batches", "skipped_steps", "first_loss",
        "final_loss", "loss_delta", "best_eval_loss", "trainable_params", "quantized",
        "device", "elapsed",
    ):
        print(f"{key:<16}{result[key]}")
    print(f"{'peak vram':<16}{gb(result['peak_vram'])} allocated, {gb(result['peak_vram_reserved'])} reserved")
    print(f"{'adapter':<16}{result['adapter_path']}")

    metrics = Path(result["adapter_path"]).parent / "metrics.jsonl"
    lines = sum(1 for _ in metrics.open(encoding="utf-8")) if metrics.exists() else 0
    print(f"{'metrics.jsonl':<16}{lines} lines")

    fell = (result["first_loss"] or 0) > (result["final_loss"] or 0)
    fits = result["peak_vram"] == 0 or result["peak_vram"] < hardware["vram_total_bytes"]
    rule("result")
    print(f"loss decreased  {ok(fell)}  {result['first_loss']} → {result['final_loss']}")
    print(f"vram in budget  {ok(fits)}")
    print(f"adapter on disk {ok(Path(result['adapter_path'], 'adapter_config.json').exists())}")
    return bool(fell and fits)


async def _free_gpu() -> list[str]:
    try:
        async with OllamaClient(read_timeout=60.0) as client:
            evicted = await client.unload_all()
    except OllamaError:
        return []
    if evicted:
        await asyncio.sleep(1.5)
    return evicted


def _print_train_event(event: dict) -> None:
    kind = event["kind"]
    if kind == "train.start":
        print(
            f"{DIM}device {event['device']} ({event.get('device_name')})  4bit={event['quantized']}  "
            f"amp={event['amp_dtype']}{RESET}"
        )
        print(
            f"{DIM}{event['trainable_params']:,} trainable of {event['total_params']:,} "
            f"({event['trainable_pct']}%)  {event['num_layers']} layers{RESET}"
        )
        data = event["data"]
        print(
            f"{DIM}data: {data['train']} train / {data['eval']} eval, "
            f"{data['label_tokens']:,} response tokens, mean len {data['mean_length']}{RESET}"
        )
        est = event["estimate"]
        print(f"{DIM}estimated peak: {gb(est['total'])} "
              f"(weights {gb(est['weights'])}, logits {gb(est['logits_and_loss'])}){RESET}")
        print(f"{DIM}{event['total_steps']} steps{RESET}\n")
    elif kind == "train.step":
        print(
            f"step {event['step']:>4}/{event['total_steps']}  loss {event['loss']:>8.4f}  "
            f"ema {event['loss_ema']:>7.4f}  lr {event['lr']:.2e}  gnorm {event['grad_norm']:>7.3f}  "
            f"{event['sec_per_step']:>5.2f}s  {event['tokens_per_second']:>6.1f} tok/s  "
            f"vram {event['vram_allocated'] / 1024 ** 3:.2f}G  eta {event['eta_seconds']:.0f}s"
        )
    elif kind == "train.eval":
        print(f"  {BOLD}eval @ {event['step']}: {event['eval_loss']:.4f}{RESET}")
    elif kind in ("train.warning", "train.retry"):
        print(f"{RED}! {event.get('message')}{RESET}")
    elif kind == "train.error":
        print(f"{RED}error: {event.get('error')}{RESET}")
    elif kind == "train.checkpoint":
        print(f"{DIM}  checkpoint → {event['path']}{RESET}")


# --------------------------------------------------------------------------------------
# 4. export
# --------------------------------------------------------------------------------------


def export(run_id: str, name: str) -> bool:
    from backend import distill, export as export_module

    if not run_id:
        runs = distill.list_runs()
        runs = [r for r in runs if r.get("has_adapter")]
        if not runs:
            print(f"{RED}No run with an adapter. Train one first.{RESET}")
            return False
        run_id = runs[0]["run_id"]

    rule(f"exporting {run_id}")
    unregister = bus.add_sink(
        lambda e: print(f"{DIM}  {e['kind']}: {e.get('step') or e.get('name') or ''}{RESET}")
        if e["kind"].startswith("export.")
        else None
    )
    try:
        result = export_module.export_run(run_id, name or f"{run_id}-export", merge=True)
    except Exception as exc:
        print(f"{RED}export failed: {type(exc).__name__}: {exc}{RESET}")
        return False
    finally:
        unregister()

    print(f"path            {result['path']}")
    print(f"size            {gb(result['size_bytes'])}")
    print(f"merged          {result['merged_path']}")
    print(f"ollama          {result['ollama_command']}")

    rule("generating from the merged model")
    prompt = "What does \"KeyError: 'name'\" mean in Python and how do I fix it?"
    print(f"{BOLD}Q{RESET} {prompt}")
    try:
        text = export_module.sample_generate(result["merged_path"], prompt, max_new_tokens=160)
    except Exception as exc:
        print(f"{RED}generation failed: {type(exc).__name__}: {exc}{RESET}")
        return False
    print(f"{BOLD}A{RESET} {text}")

    coherent = len(text.split()) >= 12
    rule("result")
    print(f"merged loads    {ok(True)}")
    print(f"generates text  {ok(coherent)}  ({len(text.split())} words)")
    print(f"Modelfile       {ok(Path(result['modelfile']).exists())}")
    return coherent


# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Ustad end-to-end verification")
    parser.add_argument("--check", action="store_true", help="hardware and Ollama preflight")
    parser.add_argument("--build", action="store_true", help="generate a real dataset")
    parser.add_argument("--train", action="store_true", help="run real QLoRA training")
    parser.add_argument("--export", action="store_true", help="merge and generate")
    parser.add_argument("--all", action="store_true", help="all four in order")
    parser.add_argument("--teacher", default="", help="Ollama tag (default: first chat-capable model)")
    parser.add_argument("--student", default="", help=f"HF repo id (default: {config.DEFAULT_STUDENT})")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--run", default="", help="run id to export (default: newest)")
    parser.add_argument("--name", default="", help="export directory name")
    parser.add_argument("--n", type=int, default=24, help="target pairs")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seq", type=int, default=0, help="override max_seq_len")
    args = parser.parse_args()

    if args.all:
        args.check = args.build = args.train = args.export = True
    if not any((args.check, args.build, args.train, args.export)):
        parser.print_help()
        return 2

    results: list[tuple[str, bool]] = []
    if args.check:
        results.append(("check", check()))
    if args.build:
        results.append(("build", build(args.teacher, args.n, args.dataset)))
    if args.train:
        results.append(("train", train(args.dataset, args.student, args.max_steps, args.seq)))
    if args.export:
        results.append(("export", export(args.run, args.name)))

    rule("summary")
    for label, passed in results:
        print(f"{label:<10}{ok(passed)}")
    return 0 if all(p for _, p in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
