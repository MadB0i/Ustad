"""LoRA / QLoRA fine-tuning of the student on the teacher's answers.

This is response-based distillation: the student is trained with plain cross-entropy on
the teacher's *text*, with the loss masked to the response tokens only. It never sees
the teacher's logits, so teacher and student need not share a tokenizer.

The loop is written by hand rather than delegated to ``transformers.Trainer``. Three
reasons, in order of weight:

* every optimiser step publishes telemetry the visualisation consumes, including
  **per-layer LoRA gradient norms**, which no callback surface exposes;
* stopping must be cooperative and immediate, mid-epoch;
* Trainer's signature moves between major versions (v5 renamed ``tokenizer`` to
  ``processing_class`` and removed ``warmup_ratio``), and this file should not care.

Runs blocking, in a worker thread. Telemetry reaches the browser through the thread-safe
event bus.
"""

from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config
from . import dataset as store
from .events import bus

_TRANSFORMERS_MAJOR = int(str(transformers.__version__).split(".")[0])

# v5 renamed the from_pretrained dtype keyword. Unknown keywords are absorbed by
# **kwargs rather than rejected, so guessing wrong would silently load fp32 instead of
# raising -- hence the version gate, plus a post-load dtype assertion below.
_DTYPE_KWARG = "dtype" if _TRANSFORMERS_MAJOR >= 5 else "torch_dtype"

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


class TrainingError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# Example construction
# --------------------------------------------------------------------------------------


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    n_label_tokens: int
    truncated: bool = False


def render_prompt(tokenizer: Any, prompt: str) -> str:
    """Render the user turn exactly as the student will see it at inference time.

    Falls back to a plain instruction format for base models with no chat template.
    ``enable_thinking=False`` is only passed when the template actually references it
    (Qwen3), so the student learns to answer directly rather than to emit ``<think>``.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return f"### Instruction:\n{prompt}\n\n### Response:\n"

    kwargs: dict[str, Any] = {}
    if "enable_thinking" in str(template):
        kwargs["enable_thinking"] = False

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    # tokenize=False returns str on both v4 and v5, but v5 made the tokenizing path
    # return a BatchEncoding, so be defensive about what comes back.
    if isinstance(rendered, str):
        return rendered
    if isinstance(rendered, (list, tuple)) and rendered and isinstance(rendered[0], str):
        return rendered[0]
    ids = rendered["input_ids"] if hasattr(rendered, "__getitem__") else None
    if ids is not None:
        flat = ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids
        return tokenizer.decode(flat, skip_special_tokens=False)
    raise TrainingError(f"Could not render a chat template (got {type(rendered).__name__}).")


def build_example(tokenizer: Any, prompt: str, response: str, max_seq_len: int) -> Example | None:
    """Tokenise one pair, masking the prompt so loss lands only on the response."""
    prefix_text = render_prompt(tokenizer, prompt)
    eos = tokenizer.eos_token or ""
    full_text = prefix_text + response + eos

    prefix_ids: list[int] = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids: list[int] = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    boundary = len(prefix_ids)
    if full_ids[:boundary] != prefix_ids:
        # BPE can merge across the prompt/response seam. Fall back to the true common
        # prefix so we never mask into the response or leave prompt tokens supervised.
        boundary = 0
        for a, b in zip(prefix_ids, full_ids):
            if a != b:
                break
            boundary += 1

    truncated = False
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        truncated = True

    # Nothing to learn from if the prompt alone fills the window.
    if boundary >= len(full_ids) - 1:
        return None

    labels = [-100] * boundary + full_ids[boundary:]
    n_label = sum(1 for token in labels if token != -100)
    if n_label < 2:
        return None
    return Example(input_ids=full_ids, labels=labels, n_label_tokens=n_label, truncated=truncated)


def collate(batch: list[Example], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(example.input_ids) for example in batch)
    input_ids, labels, attention = [], [], []
    for example in batch:
        pad = width - len(example.input_ids)
        input_ids.append(example.input_ids + [pad_id] * pad)
        labels.append(example.labels + [-100] * pad)
        attention.append([1] * len(example.input_ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def logical_numel(param: Any) -> int:
    """Parameter count as the model card would state it, not as bitsandbytes stores it.

    NF4 packs two 4-bit values into every uint8 and flattens the result, so `numel()` on a
    `Params4bit` returns half the parameters the tensor actually holds. Reporting that raw
    would understate a quantized model's size and correspondingly overstate the trainable
    percentage. `quant_state.shape` is the pre-quantization shape, so prefer it and fall
    back to doubling only if it is missing.
    """
    if param.__class__.__name__ != "Params4bit":
        return param.numel()
    shape = getattr(getattr(param, "quant_state", None), "shape", None)
    if shape is not None:
        return int(math.prod(tuple(shape)))
    return param.numel() * 2


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


@dataclass
class RunPaths:
    root: Path
    adapter: Path
    checkpoints: Path
    metrics: Path
    meta: Path

    @classmethod
    def for_run(cls, run_id: str) -> "RunPaths":
        root = config.RUNS_DIR / run_id
        paths = cls(
            root=root,
            adapter=root / "adapter",
            checkpoints=root / "checkpoints",
            metrics=root / "metrics.jsonl",
            meta=root / "run.json",
        )
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.checkpoints.mkdir(parents=True, exist_ok=True)
        return paths


class TrainingRun:
    def __init__(self, tc: config.TrainConfig, run_id: str | None = None) -> None:
        self.tc = tc
        self.run_id = run_id or time.strftime("run-%Y%m%d-%H%M%S")
        self.paths = RunPaths.for_run(self.run_id)
        self._stop = threading.Event()
        self.result: dict[str, Any] = {}

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    # -- telemetry ---------------------------------------------------------------

    def _emit(self, kind: str, **payload: Any) -> None:
        event = bus.publish(kind, run_id=self.run_id, **payload)
        if kind in ("train.step", "train.eval"):
            # Persisted so the UI can rebuild its charts after a refresh, and so a run
            # remains inspectable long after the process exits.
            with self.paths.metrics.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    # -- entry point -------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Train, retrying once at half the sequence length if the GPU runs out."""
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._execute()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                if attempts >= 2 or self.tc.max_seq_len <= 128:
                    self._emit(
                        "train.error",
                        error=(
                            f"Out of GPU memory at seq_len={self.tc.max_seq_len}. "
                            "Lower the sequence length, the LoRA rank, or the batch size."
                        ),
                        detail=str(exc)[:400],
                        oom=True,
                    )
                    raise TrainingError("out of GPU memory") from exc
                previous = self.tc.max_seq_len
                self.tc.max_seq_len = max(128, previous // 2)
                self._emit(
                    "train.retry",
                    message=(
                        f"Out of GPU memory at seq_len={previous}; retrying at "
                        f"{self.tc.max_seq_len}."
                    ),
                    max_seq_len=self.tc.max_seq_len,
                )
            except Exception as exc:
                self._emit("train.error", error=str(exc)[:600], detail=type(exc).__name__)
                raise

    # -- setup -------------------------------------------------------------------

    def _seed(self) -> None:
        random.seed(self.tc.seed)
        torch.manual_seed(self.tc.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.tc.seed)

    def _load_tokenizer(self) -> Any:
        tokenizer = AutoTokenizer.from_pretrained(self.tc.student, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self, hardware: dict[str, Any]) -> tuple[Any, bool]:
        use_4bit = bool(self.tc.load_in_4bit and hardware["tier"].startswith("gpu"))
        kwargs: dict[str, Any] = {"trust_remote_code": False}

        if use_4bit:
            from transformers import BitsAndBytesConfig

            # bf16 needs Ampere; on Turing and older the compute dtype must be fp16.
            compute_dtype = torch.bfloat16 if hardware["supports_bf16"] else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            kwargs["device_map"] = {"": 0}
            kwargs[_DTYPE_KWARG] = compute_dtype
        elif hardware["cuda_available"] and hardware["arch_supported"]:
            kwargs["device_map"] = {"": 0}
            kwargs[_DTYPE_KWARG] = torch.float16
        else:
            kwargs["device_map"] = {"": "cpu"}
            kwargs[_DTYPE_KWARG] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.tc.student, **kwargs)
        model.config.use_cache = False  # required with gradient checkpointing
        return model, use_4bit

    def _attach_lora(self, model: Any, use_4bit: bool) -> Any:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if use_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.tc.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        elif self.tc.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()

        present = {name.split(".")[-1] for name, _ in model.named_modules()}
        targets = [t for t in self.tc.target_modules if t in present]
        if not targets:
            raise TrainingError(
                f"None of the target modules {self.tc.target_modules} exist in "
                f"{self.tc.student}. Adjust the target module list."
            )
        missing = sorted(set(self.tc.target_modules) - set(targets))
        if missing:
            self._emit("train.warning", message=f"target modules not present, skipped: {', '.join(missing)}")

        peft_model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=self.tc.lora_r,
                lora_alpha=self.tc.lora_alpha,
                lora_dropout=self.tc.lora_dropout,
                target_modules=targets,
                bias="none",
            ),
        )
        return peft_model

    def _build_examples(self, tokenizer: Any) -> tuple[list[Example], list[Example], dict[str, Any]]:
        pairs = store.load_pairs(self.tc.dataset)
        if not pairs:
            raise TrainingError(f"Dataset '{self.tc.dataset}' has no usable rows.")

        examples: list[Example] = []
        skipped = truncated = 0
        for pair in pairs:
            example = build_example(tokenizer, pair["prompt"], pair["response"], self.tc.max_seq_len)
            if example is None:
                skipped += 1
                continue
            truncated += int(example.truncated)
            examples.append(example)
        if not examples:
            raise TrainingError("Every row was longer than the sequence window; raise max_seq_len.")

        rng = random.Random(self.tc.seed)
        rng.shuffle(examples)
        n_eval = 0
        if self.tc.eval_ratio > 0 and len(examples) >= 10:
            n_eval = max(1, int(len(examples) * self.tc.eval_ratio))
        eval_set = examples[:n_eval]
        train_set = examples[n_eval:]

        stats = {
            "pairs": len(pairs),
            "examples": len(examples),
            "train": len(train_set),
            "eval": len(eval_set),
            "skipped": skipped,
            "truncated": truncated,
            "label_tokens": sum(e.n_label_tokens for e in train_set),
            "mean_length": round(sum(len(e.input_ids) for e in examples) / len(examples), 1),
        }
        return train_set, eval_set, stats

    # -- the loop ----------------------------------------------------------------

    def _execute(self) -> dict[str, Any]:
        started = time.time()
        self._seed()
        hardware = config.detect_hardware()

        if hardware["tier"] == "cpu" and self.tc.load_in_4bit:
            self.tc.load_in_4bit = False
            self._emit("train.warning", message="No usable CUDA device; training on CPU in fp32.")

        on_gpu = hardware["tier"].startswith("gpu")
        device = torch.device("cuda:0" if on_gpu else "cpu")
        # fp16 autocast needs loss scaling to keep small gradients from flushing to zero.
        # bf16 has fp32's exponent range, so it does not.
        use_amp = on_gpu
        amp_dtype = torch.bfloat16 if (on_gpu and hardware["supports_bf16"]) else torch.float16
        needs_scaler = use_amp and amp_dtype is torch.float16

        self._emit(
            "train.loading",
            message=f"loading {self.tc.student}",
            student=self.tc.student,
            device=str(device),
            tier=hardware["tier"],
        )

        tokenizer = self._load_tokenizer()
        train_set, eval_set, data_stats = self._build_examples(tokenizer)

        model, use_4bit = self._load_model(hardware)
        model_config = model.config.to_dict()
        model = self._attach_lora(model, use_4bit)
        model.train()

        trainable = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(logical_numel(p) for p in trainable)
        n_total = sum(logical_numel(p) for p in model.parameters())
        layer_groups = self._group_by_layer(model)
        n_layers = int(model_config.get("num_hidden_layers", 0)) or (max(layer_groups) + 1 if layer_groups else 0)

        batches = self._plan_batches(train_set)
        steps_per_epoch = max(1, math.ceil(len(batches) / self.tc.grad_accum))
        total_steps = self.tc.max_steps or max(1, int(steps_per_epoch * self.tc.epochs))

        # A small dataset makes for a short run: 60 pairs at accum 8 is ~21 steps, and a fixed
        # eval_every of 25 would then never fire, leaving the loss chart with a single final
        # point. Scale the cadence down to give roughly four eval points on any run length,
        # and never up -- a long run keeps the configured interval.
        eval_every = min(self.tc.eval_every, max(1, total_steps // 4)) if self.tc.eval_every > 0 else 0

        optimizer = torch.optim.AdamW(
            trainable, lr=self.tc.lr, weight_decay=self.tc.weight_decay, betas=(0.9, 0.999), eps=1e-8
        )
        warmup = max(1, int(total_steps * self.tc.warmup_ratio)) if self.tc.warmup_ratio > 0 else 0
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: self._lr_multiplier(step, warmup, total_steps)
        )
        scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

        self._emit(
            "train.start",
            student=self.tc.student,
            dataset=self.tc.dataset,
            device=str(device),
            device_name=hardware.get("device_name"),
            tier=hardware["tier"],
            quantized=use_4bit,
            amp_dtype=str(amp_dtype).replace("torch.", "") if use_amp else "fp32",
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
            epochs=self.tc.epochs,
            eval_every=eval_every,
            num_layers=n_layers,
            lora_layers=sorted(layer_groups),
            trainable_params=n_trainable,
            total_params=n_total,
            trainable_pct=round(100 * n_trainable / max(1, n_total), 4),
            data=data_stats,
            estimate=config.estimate_peak_vram(model_config, self.tc),
            config=self.tc.to_dict(),
        )

        pad_id = tokenizer.pad_token_id or 0
        history: list[float] = []
        step_times: list[float] = []
        step = 0
        micro = 0
        skipped_steps = 0
        best_eval = math.inf
        stopped_early = False
        step_started = time.time()
        window_loss = 0.0
        window_tokens = 0

        if on_gpu:
            torch.cuda.reset_peak_memory_stats()

        epoch_count = max(1, math.ceil(total_steps / steps_per_epoch))
        for epoch in range(epoch_count):
            if step >= total_steps or self.stopping:
                break
            if epoch > 0:
                batches = self._plan_batches(train_set, epoch)

            for window in self._accumulation_windows(batches):
                if step >= total_steps or self.stopping:
                    stopped_early = self.stopping
                    break

                # Token-weighted accumulation: each micro-batch contributes in
                # proportion to how many response tokens it supervises, so a one-line
                # answer does not carry the same weight as a long one.
                window_total = sum(sum(e.n_label_tokens for e in batch) for batch in window) or 1
                optimizer.zero_grad(set_to_none=True)
                window_loss = 0.0
                window_tokens = 0

                for batch in window:
                    if self.stopping:
                        stopped_early = True
                        break
                    tensors = {k: v.to(device, non_blocking=True) for k, v in collate(batch, pad_id).items()}
                    batch_tokens = sum(e.n_label_tokens for e in batch)

                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        output = model(**tensors)
                        loss = output.loss

                    scaled = loss * (batch_tokens / window_total)
                    scaler.scale(scaled).backward() if needs_scaler else scaled.backward()

                    window_loss += float(loss.detach()) * batch_tokens
                    window_tokens += batch_tokens
                    micro += 1

                if window_tokens == 0:
                    continue

                if needs_scaler:
                    scaler.unscale_(optimizer)

                # Read per-layer norms after unscaling but before clipping, so they are
                # the true gradient magnitudes the visualisation claims to show.
                layer_norms = self._layer_norms(layer_groups, n_layers)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable, self.tc.max_grad_norm)
                )

                scale_before = scaler.get_scale() if needs_scaler else 1.0
                if needs_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                    # A lowered scale means inf/NaN gradients were found and the
                    # optimiser step was skipped; the LR schedule should not advance.
                    if scaler.get_scale() < scale_before:
                        skipped_steps += 1
                    else:
                        scheduler.step()
                else:
                    optimizer.step()
                    scheduler.step()

                step += 1
                now = time.time()
                elapsed = now - step_started
                step_started = now
                step_times.append(elapsed)
                mean_loss = window_loss / window_tokens
                history.append(mean_loss)

                if step % max(1, self.tc.log_every) == 0 or step == total_steps:
                    recent = step_times[-20:]
                    per_step = sum(recent) / len(recent)
                    self._emit(
                        "train.step",
                        step=step,
                        total_steps=total_steps,
                        epoch=round(step / steps_per_epoch, 3),
                        loss=round(mean_loss, 5),
                        loss_ema=round(_ema(history), 5),
                        lr=scheduler.get_last_lr()[0],
                        grad_norm=round(grad_norm, 5),
                        label_tokens=window_tokens,
                        tokens_per_second=round(window_tokens / elapsed, 1) if elapsed > 0 else 0.0,
                        sec_per_step=round(elapsed, 3),
                        eta_seconds=round(per_step * (total_steps - step), 1),
                        skipped=skipped_steps,
                        vram_allocated=int(torch.cuda.memory_allocated()) if on_gpu else 0,
                        vram_peak=int(torch.cuda.max_memory_allocated()) if on_gpu else 0,
                        vram_reserved=int(torch.cuda.memory_reserved()) if on_gpu else 0,
                    )
                    self._emit("train.layer_grads", step=step, norms=layer_norms)

                if eval_set and eval_every > 0 and step % eval_every == 0:
                    eval_loss = self._evaluate(model, eval_set, pad_id, device, use_amp, amp_dtype)
                    best_eval = min(best_eval, eval_loss)
                    self._emit("train.eval", step=step, eval_loss=round(eval_loss, 5), best=round(best_eval, 5))
                    model.train()

                if self.tc.save_every > 0 and step % self.tc.save_every == 0 and step < total_steps:
                    target = self.paths.checkpoints / f"step-{step}"
                    model.save_pretrained(str(target))
                    self._emit("train.checkpoint", step=step, path=str(target))

            if stopped_early:
                break

        if eval_set:
            final_eval = self._evaluate(model, eval_set, pad_id, device, use_amp, amp_dtype)
            best_eval = min(best_eval, final_eval)
            self._emit("train.eval", step=step, eval_loss=round(final_eval, 5), best=round(best_eval, 5), final=True)

        model.save_pretrained(str(self.paths.adapter))
        tokenizer.save_pretrained(str(self.paths.adapter))

        result = {
            "run_id": self.run_id,
            "student": self.tc.student,
            "dataset": self.tc.dataset,
            "steps_completed": step,
            "total_steps": total_steps,
            "micro_batches": micro,
            "skipped_steps": skipped_steps,
            "first_loss": round(history[0], 5) if history else None,
            "final_loss": round(history[-1], 5) if history else None,
            "best_eval_loss": round(best_eval, 5) if best_eval < math.inf else None,
            "loss_delta": round(history[0] - history[-1], 5) if len(history) > 1 else None,
            "stopped_early": stopped_early,
            "adapter_path": str(self.paths.adapter),
            "trainable_params": n_trainable,
            "quantized": use_4bit,
            "device": str(device),
            "peak_vram": int(torch.cuda.max_memory_allocated()) if on_gpu else 0,
            "peak_vram_reserved": int(torch.cuda.max_memory_reserved()) if on_gpu else 0,
            "elapsed": round(time.time() - started, 1),
            "data": data_stats,
            "config": self.tc.to_dict(),
        }
        self.paths.meta.write_text(json.dumps(result, indent=2), encoding="utf-8")
        self._emit("train.done", **result)
        self.result = result

        del model
        if on_gpu:
            torch.cuda.empty_cache()
        return result

    # -- helpers -----------------------------------------------------------------

    @staticmethod
    def _lr_multiplier(step: int, warmup: int, total: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _plan_batches(self, examples: list[Example], epoch: int = 0) -> list[list[Example]]:
        order = list(range(len(examples)))
        random.Random(self.tc.seed + epoch).shuffle(order)
        size = max(1, self.tc.batch_size)
        return [[examples[i] for i in order[start : start + size]] for start in range(0, len(order), size)]

    def _accumulation_windows(self, batches: list[list[Example]]) -> Iterator[list[list[Example]]]:
        accum = max(1, self.tc.grad_accum)
        for start in range(0, len(batches), accum):
            yield batches[start : start + accum]

    @staticmethod
    def _group_by_layer(model: Any) -> dict[int, list[torch.nn.Parameter]]:
        """Map transformer layer index -> its trainable LoRA parameters."""
        groups: dict[int, list[torch.nn.Parameter]] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            match = _LAYER_RE.search(name)
            if match:
                groups.setdefault(int(match.group(1)), []).append(param)
        return groups

    @staticmethod
    def _layer_norms(groups: dict[int, list[torch.nn.Parameter]], n_layers: int) -> list[float]:
        """L2 norm of each layer's LoRA gradients. Cheap: these are small tensors."""
        width = n_layers or ((max(groups) + 1) if groups else 0)
        norms = [0.0] * width
        for index, params in groups.items():
            if index >= width:
                continue
            total = 0.0
            for param in params:
                if param.grad is not None:
                    total += float(param.grad.detach().float().pow(2).sum())
            norms[index] = round(math.sqrt(total), 6)
        return norms

    def _evaluate(
        self,
        model: Any,
        examples: list[Example],
        pad_id: int,
        device: torch.device,
        use_amp: bool,
        amp_dtype: torch.dtype,
    ) -> float:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        size = max(1, self.tc.batch_size)
        with torch.no_grad():
            for start in range(0, len(examples), size):
                batch = examples[start : start + size]
                tensors = {k: v.to(device) for k, v in collate(batch, pad_id).items()}
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    loss = model(**tensors).loss
                tokens = sum(e.n_label_tokens for e in batch)
                total_loss += float(loss.detach()) * tokens
                total_tokens += tokens
        return total_loss / max(1, total_tokens)


def _ema(values: list[float], span: int = 12) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1)
    out = values[0]
    for value in values[1:]:
        out = alpha * value + (1 - alpha) * out
    return out


# --------------------------------------------------------------------------------------
# Run inspection
# --------------------------------------------------------------------------------------


def list_runs() -> list[dict[str, Any]]:
    runs = []
    for directory in sorted(config.RUNS_DIR.glob("run-*"), reverse=True):
        meta_path = directory / "run.json"
        entry: dict[str, Any] = {"run_id": directory.name, "has_adapter": (directory / "adapter").exists()}
        if meta_path.exists():
            try:
                entry.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        runs.append(entry)
    return runs


def read_run(run_id: str) -> dict[str, Any]:
    directory = config.RUNS_DIR / store.safe_name(run_id)
    if not directory.is_dir():
        raise FileNotFoundError(run_id)
    meta: dict[str, Any] = {}
    if (directory / "run.json").exists():
        meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    metrics: list[dict[str, Any]] = []
    metrics_path = directory / "metrics.jsonl"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return {"run_id": directory.name, "meta": meta, "metrics": metrics}
