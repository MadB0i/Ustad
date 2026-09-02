# Ustad

A local-first studio for **distilling** a large open-weight LLM into a small one you own.

You pick a model Ollama has already pulled — that becomes the **teacher**. You describe a
skill and give a handful of seed prompts. The teacher writes more prompts like them,
answers them all, and those (prompt, response) pairs become a local JSONL dataset. A small
**student** model is then LoRA/QLoRA fine-tuned on that dataset, and you get back a LoRA
adapter plus a merged model you can run in Ollama.

Everything after the one-time model downloads runs on your machine. No API keys, no cloud
inference, no telemetry.

```
seed prompts ──► teacher self-instructs ──► teacher answers ──► pairs.jsonl
                    (Ollama, local)          (Ollama, local)         │
                                                                     ▼
                    LoRA adapter + merged model ◄── QLoRA fine-tune (transformers + peft)
```

This is **response-based** distillation: the student learns from the teacher's *outputs*,
never its logits or weights. That is what makes it work across families — a Qwen2.5 teacher
can train a TinyLlama student, because no tokenizer or architecture alignment is required.

---

## Requirements

| | |
|---|---|
| **Python** | 3.10–3.12 (torch has no 3.13+ wheels yet) |
| **Ollama** | installed and running, with at least one instruction-tuned model pulled |
| **GPU** | optional. 4 GB+ NVIDIA with compute capability ≥ 6.0 for 4-bit QLoRA |
| **CPU fallback** | works, in fp32 at a shorter sequence length, and is much slower |
| **Disk** | ~5 GB for a teacher + a student + one merged export |

The teacher must be a **chat/instruction** model. Ustad probes each model's Ollama template
and greys out completion-only models rather than letting a run fail after ten minutes.

## Install

```powershell
git clone <this repo> Ustad; cd Ustad; .\scripts\quickstart.ps1
```

```bash
git clone <this repo> Ustad && cd Ustad && ./scripts/quickstart.sh
```

The quickstart finds a suitable Python, creates `.venv`, installs the right torch build for
your machine, starts the Ollama daemon if it isn't listening, and launches the app at
<http://127.0.0.1:8177>.

Manually, if you prefer:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m backend.server
```

`torch` is deliberately **not** in `requirements.txt` — the wheel you need depends on your
GPU, and pinning a CUDA build there would break every CPU-only install. Use
`.../whl/cpu` instead of `.../whl/cu126` if you have no NVIDIA card.

> **Why cu126?** It is the most recent PyTorch index that still ships kernels for every
> arch from Maxwell to Hopper. Newer indexes drop older ones; a wheel without your arch
> fails at the first matmul with "no kernel image is available", long after everything has
> claimed to be fine. `smoke_test.py --check` proves the kernels exist by *running* one.

## Using it

1. **Teacher** — pick a pulled model. Badges tell you whether it fits your VRAM, whether
   it's already resident, and whether it's a thinking model (`<think>` blocks are stripped
   automatically).
2. **Skill + seeds** — describe the behaviour you want in a sentence, then give 4–8 real
   example prompts. These are the only human-written text in the pipeline, so they set the
   ceiling on quality. Set a target pair count and press **Build dataset**.
3. **Review** — the Dataset drawer pages through what the teacher produced. Delete bad
   rows before training. A truncated or refused answer teaches the student to truncate and
   refuse, so this step matters more than it looks.
4. **Student** — pick a base model, check the VRAM estimate, press **Start training**.
   Ustad evicts the teacher from the GPU first (`keep_alive: 0`) so the two never contend.
5. **Export** — pick the run, merge, and copy the printed `ollama create` command.

## What the visualization actually shows

The centre and right panels are a **readout**, not an ornament. Every moving pixel is
downstream of a real backend event delivered over SSE:

| On screen | Real signal behind it |
|---|---|
| Amber pulse on a teacher model | that model is mid-generation right now |
| Streaming text under the teacher | actual tokens off Ollama's `/api/chat` stream |
| A packet crossing the channel | one dataset pair written, or one optimiser step taken |
| Brightness of a layer row | that layer's **LoRA gradient norm** this step |
| Row size / edge glow | same norm, √-scaled against the running peak |
| The downward sweep | one real optimiser step, travelling the way the backward pass does |
| Loss line + smoothed line | per-step training loss and its EMA |
| Diamonds on the loss chart | held-out eval loss |
| step / loss / lr / grad-norm / tok-s / VRAM / ETA | read straight off the training loop |

Gradient norms are read **after** `scaler.unscale_` and **before** clipping, so they are the
true gradient magnitudes rather than scaled or clipped stand-ins. When nothing is running,
nothing moves — there is no idle animation and no synthetic data anywhere in the frontend.

`prefers-reduced-motion` disables the packets, the sweep, and the pulse; the numbers and the
loss chart keep updating.

## Command-line verification

The UI is optional. Everything is reachable from one script, and none of it is mocked:

```bash
python scripts/smoke_test.py --check                 # hardware + Ollama preflight
python scripts/smoke_test.py --build --n 24          # generate a real dataset
python scripts/smoke_test.py --train --max-steps 30  # real QLoRA training
python scripts/smoke_test.py --export                # merge, then generate from the result
python scripts/smoke_test.py --all                   # all four, in order
```

`--check` is worth running on any new machine: it prints your compute capability,
`torch.cuda.get_arch_list()`, free VRAM, the chosen preset, and then actually executes an
fp16 matmul and a `bitsandbytes` NF4 layer on the GPU.

## Exporting

An export directory contains:

```
adapter/       the LoRA adapter alone (a few MB) — use with peft on top of the base model
merged/        base + adapter merged in fp16, a normal HF model directory
Modelfile      ready for: ollama create <name> -f Modelfile
README.md      the run's provenance: teacher, student, dataset, steps, losses
export.json    the same as machine-readable metadata
```

To run it in Ollama:

```bash
cd data/exports/<name> && ollama create my-student -f Modelfile
```

Ollama's `FROM ./merged` reads the safetensors directory directly. For a smaller quantized
GGUF, the export README prints the exact `convert_hf_to_gguf.py` and `llama-quantize`
commands for your paths.

Merging happens **on the CPU in fp16**, never into the 4-bit weights — merging a LoRA into
NF4 weights would fold the adapter into an already-lossy tensor and quietly degrade the
model. The base is reloaded unquantized for the merge, which is why it needs RAM but no VRAM.

## Layout

```
backend/
  config.py         paths, hardware detection, device tiers, LoRA presets, VRAM estimator
  events.py         thread-safe event bus with replay, fanned out over SSE
  ollama_client.py  async httpx wrapper: tags, ps, show, chat-stream, unload, health
  dataset.py        self-instruct expansion → teacher answers → filtered JSONL
  distill.py        QLoRA load + hand-written training loop + telemetry
  jobs.py           single-slot job manager (one GPU consumer at a time)
  export.py         CPU fp16 merge + Modelfile + GGUF commands
  server.py         FastAPI REST + SSE + static mount
frontend/           index.html, app.js, style.css — no build step
scripts/            quickstart.ps1, quickstart.sh, smoke_test.py
data/               hf-cache/  datasets/  runs/<run_id>/  exports/<name>/   (gitignored)
```

`data/` holds everything mutable, including the Hugging Face cache — `HF_HOME` is
repointed there by `backend/config.py` at import time, so models land next to the project
instead of on your system drive. Nothing outside `data/` is written at runtime.

## Design notes

**Why a hand-written training loop instead of `Trainer`.** Three reasons, all load-bearing:
per-step SSE telemetry with per-layer gradient norms, a cooperative stop that still saves
the adapter, and response-only loss masking. A stock `Trainer` would need callbacks and
subclassing for each, and transformers v5 moved several of the relevant keywords. ~120 lines
of explicit loop is less surface than that.

**Response-only loss.** The prompt prefix is masked to `-100` so the student is scored only
on the teacher's answer, not on reproducing the question. The prefix boundary is found by
templating the user turn alone — and then verified token-by-token, because BPE can merge
across the prompt/response seam and a naive prefix length would either mask into the
response or supervise part of the prompt.

**Token-weighted gradient accumulation.** The usual `loss / grad_accum` weights every
micro-batch equally, which over-weights short answers when response lengths vary — and with
response-only masking they vary a lot. Each micro-batch is instead weighted by its share of
the accumulation window's label tokens, which the loop can do exactly because it owns its
own batching.

**fp16 + GradScaler, not bf16.** `torch.cuda.is_bf16_supported()` returns `True` on Turing,
because it counts software emulation. There are no bf16 tensor cores before Ampere, so
Ustad gates on compute capability ≥ 8.0 and uses fp16 with loss scaling below that.
Skipped steps (from a scaler overflow) are detected and don't advance the LR schedule.

**One GPU consumer at a time.** Ollama keeps a teacher resident for five minutes by
default, so training started right after generation would OOM on a small card. Dataset
builds, training, and exports all pass through a single-slot job manager, and training
explicitly evicts the teacher and waits for the driver to actually release the allocation.

**Where the VRAM goes.** For small models with large vocabularies the LM head dominates,
not the weights. Qwen3-0.6B has a 151,936-token tied vocab: at seq 512 the fp16 logits, the
fp32 cross-entropy upcast, and their gradients cost more than the 4-bit weights do. That is
why the 4 GB preset is seq 512 / batch 1 with accumulation, rather than a larger batch —
and why the estimate shown before a run breaks out `logits_and_loss` separately.

## Troubleshooting

**"Ollama unreachable"** — the daemon isn't listening on 11434. Run `ollama serve`. Ustad
never starts or stops it during a session; only the quickstart will start it.

**A model shows "no chat template"** — it's a base/completion model. It will produce
continuations rather than answers. Pull an instruct model instead.

**Out of GPU memory** — the run reports it explicitly and retries once at half the sequence
length. If it still fails: lower seq length, then LoRA rank, then turn off 4-bit only if you
have the headroom (4-bit *saves* memory). Check nothing else is holding the GPU with
`ollama ps`.

**Training is slow on CPU** — expected. The CPU preset drops to seq 256, rank 8, and no
gradient checkpointing (checkpointing trades compute for memory, which is the wrong trade
without VRAM pressure). Use `--max-steps` to keep runs bounded.

**Loss is flat** — usually the dataset, not the hyperparameters. Check the Dataset drawer
for truncated or refused answers, and check that your seeds actually demonstrate the skill.
20 good pairs beat 200 mediocre ones.

**Training loss falls but eval loss rises** — you are past the useful point. Watch the gap
between the two, not the training loss alone. This bites hardest on small datasets, because
*steps* and *epochs* come apart: 22 rows at accumulation 8 is only ~3 optimiser steps per
epoch, so `--max-steps 30` is about **ten** epochs, not thirty steps' worth of learning.
Leave **Max steps** at 0 and control the run with **Epochs** unless you are deliberately
cutting a run short.

## Limits

Text LLMs only — no vision, audio, or multimodal. Single user, loopback-bound, no auth,
because there is no network surface to authenticate. One job at a time by design. Distilled
students inherit the teacher's mistakes: this transfers *style and format* reliably, and
factual accuracy only as far as the teacher had it.
#   U s t a d  
 