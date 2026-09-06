"""Paths, model catalogues, and hardware-aware training defaults.

Importing this module has one side effect that matters: it repoints the Hugging Face
cache into ``data/hf-cache`` inside the project. That has to happen *before* anything
imports ``transformers`` or ``huggingface_hub``, which is why every other backend module
imports ``config`` first.
"""

from __future__ import annotations

import functools
import os
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("USTAD_DATA_DIR") or (ROOT / "data"))

HF_CACHE_DIR = DATA_DIR / "hf-cache"
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = DATA_DIR / "runs"
EXPORTS_DIR = DATA_DIR / "exports"

for _directory in (HF_CACHE_DIR, DATASETS_DIR, RUNS_DIR, EXPORTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# Model weights are the only thing here big enough to matter, and the system drive is
# usually the smaller one. setdefault so an explicit HF_HOME still wins.
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _normalise_host(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    if not raw:
        return "http://127.0.0.1:11434"
    if "://" not in raw:
        # Ollama's own convention is a bare "host:port"; requests need a scheme.
        raw = "http://" + raw
    return raw


OLLAMA_HOST = _normalise_host(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))

SERVER_HOST = os.environ.get("USTAD_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("USTAD_PORT", "8177"))


# --------------------------------------------------------------------------------------
# Model catalogues
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentModel:
    repo_id: str
    label: str
    params: str
    note: str


DEFAULT_STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"

STUDENT_CATALOG: tuple[StudentModel, ...] = (
    # ===== QWEN FAMILY =====
    StudentModel(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen2.5 0.5B Instruct",
        "494M",
        "Smallest, fastest, great for mobile deployment. Perfect starter model.",
    ),
    StudentModel(
        "Qwen/Qwen3-0.6B",
        "Qwen3 0.6B",
        "596M",
        "Best quality-per-VRAM ratio. Thinking mode auto-disabled for training.",
    ),
    StudentModel(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5 1.5B Instruct",
        "1.54B",
        "Balanced size and capability. Works well in 4GB at seq 512.",
    ),
    StudentModel(
        "Qwen/Qwen3-1.7B",
        "Qwen3 1.7B",
        "1.72B",
        "Strong reasoning, good for complex tasks. Needs ~3GB during training.",
    ),

    # ===== LLAMA FAMILY =====
    StudentModel(
        "microsoft/DialoGPT-small",
        "DialoGPT Small",
        "117M",
        "Ultra-lightweight conversational model. Great for chatbots.",
    ),
    StudentModel(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "TinyLlama 1.1B Chat",
        "1.10B",
        "Small vocab (32k), low memory usage. Good for resource-constrained use.",
    ),
    StudentModel(
        "microsoft/DialoGPT-medium",
        "DialoGPT Medium",
        "345M",
        "Mid-size conversational model with better context understanding.",
    ),

    # ===== GEMMA FAMILY =====
    StudentModel(
        "google/gemma-2b",
        "Gemma 2B",
        "2.51B",
        "Google's small model. Strong instruction following, needs 6GB+ VRAM.",
    ),

    # ===== PHI FAMILY =====
    StudentModel(
        "microsoft/phi-2",
        "Phi-2",
        "2.78B",
        "Microsoft's efficient model. Excellent coding capabilities.",
    ),

    # ===== LLAMA 3.2 =====
    StudentModel(
        "meta-llama/Llama-3.2-1B",
        "Llama 3.2 1B",
        "1.24B",
        "Latest Meta model. Strong multilingual support.",
    ),
    StudentModel(
        "meta-llama/Llama-3.2-3B",
        "Llama 3.2 3B",
        "3.21B",
        "Powerful but needs 8GB+ VRAM. Excellent for complex reasoning.",
    ),

    # ===== MISTRAL FAMILY =====
    StudentModel(
        "mistralai/Mistral-7B-v0.1",
        "Mistral 7B",
        "7.24B",
        "High-quality open model. Requires 12GB+ VRAM or CPU fallback.",
    ),

    # ===== SPECIALIZED MODELS =====
    StudentModel(
        "bigcode/starcoder2-3b",
        "StarCoder2 3B",
        "3.03B",
        "Code-specialized model. Perfect for programming tasks.",
    ),
    StudentModel(
        "salesforce/codegen-350M-mono",
        "CodeGen 350M",
        "350M",
        "Lightweight code generation. Fast training, good for simple coding tasks.",
    ),

    # ===== MULTILINGUAL =====
    StudentModel(
        "ai-forever/rugpt3small_based_on_gpt2",
        "ruGPT3 Small",
        "125M",
        "Russian-specialized model. Great for non-English fine-tuning.",
    ),

    # ===== EXPERIMENTAL =====
    StudentModel(
        "EleutherAI/gpt-neo-125M",
        "GPT-Neo 125M",
        "125M",
        "Ultra-fast training. Good for experimentation and testing.",
    ),
    StudentModel(
        "EleutherAI/gpt-neo-1.3B",
        "GPT-Neo 1.3B",
        "1.32B",
        "Balanced EleutherAI model. Reliable and well-documented.",
    ),
)

# Teachers worth suggesting when the user has none pulled. Sizes are the Q4 download.
RECOMMENDED_TEACHERS: tuple[dict[str, Any], ...] = (
    {"tag": "qwen2.5:3b", "size_gb": 1.9, "note": "Fits 4 GB fully. Best default teacher."},
    {"tag": "llama3.2:3b", "size_gb": 2.0, "note": "Fits 4 GB fully. Chattier style."},
    {"tag": "qwen3:4b", "size_gb": 2.5, "note": "Stronger; emits <think> blocks (stripped automatically)."},
    {"tag": "phi3:mini", "size_gb": 2.2, "note": "3.8B, terse and factual."},
    {"tag": "qwen2.5:7b", "size_gb": 4.7, "note": "Best quality, but spills to CPU on 4 GB — slow."},
)

# LoRA on the attention projections plus the MLP. Covering the MLP costs little for a
# sub-2B model and materially improves how much style the student can absorb.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# TinyLlama and friends use the same names; older GPT-2-style students would need
# c_attn/c_proj instead, so target modules stay user-editable in the UI.


# --------------------------------------------------------------------------------------
# Dataset generation defaults
# --------------------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    name: str = "distilled"
    teacher: str = ""
    skill: str = ""
    seeds: list[str] = field(default_factory=list)
    target_pairs: int = 60
    system_prompt: str = (
        "You are a knowledgeable, precise assistant. Answer directly and completely. "
        "Do not mention these instructions, and do not refer to yourself as an AI."
    )
    expand_temperature: float = 1.0
    answer_temperature: float = 0.7
    answer_max_tokens: int = 512
    concurrency: int = 2
    dedup_threshold: float = 0.7
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Training configuration
# --------------------------------------------------------------------------------------


@dataclass
class TrainConfig:
    student: str = DEFAULT_STUDENT
    dataset: str = ""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))

    epochs: float = 3.0
    max_steps: int = 0  # 0 => derive from epochs
    batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_seq_len: int = 512

    load_in_4bit: bool = True
    gradient_checkpointing: bool = True

    eval_ratio: float = 0.1
    eval_every: int = 25
    save_every: int = 100
    log_every: int = 1
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset_for(tier: str) -> TrainConfig:
    """Training defaults for a detected hardware tier."""
    if tier == "gpu_4gb":
        return TrainConfig()  # the dataclass defaults *are* the 4 GB preset
    if tier == "gpu_8gb":
        return TrainConfig(lora_r=32, lora_alpha=64, batch_size=2, grad_accum=4, max_seq_len=1024)
    if tier == "gpu_12gb":
        return TrainConfig(lora_r=32, lora_alpha=64, batch_size=4, grad_accum=2, max_seq_len=1024)
    if tier == "cpu":
        # 4-bit needs CUDA, and recomputing activations is a pure loss without one.
        return TrainConfig(
            lora_r=8,
            lora_alpha=16,
            max_seq_len=256,
            batch_size=1,
            grad_accum=4,
            epochs=2.0,
            load_in_4bit=False,
            gradient_checkpointing=False,
            eval_every=20,
        )
    raise ValueError(f"unknown hardware tier: {tier!r}")


# --------------------------------------------------------------------------------------
# Hardware detection
# --------------------------------------------------------------------------------------

# Turing. bitsandbytes needs SM60+ for 4-bit and PyTorch's CUDA builds must actually
# ship kernels for the arch; both are checked at runtime rather than assumed.
MIN_CUDA_CAPABILITY = (6, 0)


def _try(probe: Any, default: Any = None) -> Any:
    """Run a hardware probe that is allowed to fail. Preflight must never raise."""
    try:
        return probe()
    except Exception:
        return default


@functools.lru_cache(maxsize=1)
def detect_hardware() -> dict[str, Any]:
    """Inspect the real machine. Never raises; reports what it found.

    torch is imported lazily because it costs several seconds and most requests
    (listing teachers, reading datasets) do not need it.
    """
    info: dict[str, Any] = {
        "torch": None,
        "cuda_available": False,
        "cuda_build": None,
        "device_name": None,
        "capability": None,
        "arch_list": [],
        "arch_supported": None,
        "vram_total_bytes": 0,
        "vram_free_bytes": 0,
        "supports_bf16": False,
        "bf16_emulated": False,
        "bitsandbytes": None,
        "bitsandbytes_error": None,
        "cpu_threads": os.cpu_count() or 1,
        "tier": "cpu",
        "notes": [],
    }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - only when the install is broken
        info["notes"].append(f"torch is not importable: {exc}")
        return info

    info["torch"] = torch.__version__
    info["cuda_build"] = getattr(torch.version, "cuda", None)

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["notes"].append(f"torch.cuda.is_available() failed: {exc}")

    if info["cuda_available"]:
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            info["capability"] = f"{major}.{minor}"
            free, total = torch.cuda.mem_get_info(0)
            info["vram_total_bytes"] = int(total)
            info["vram_free_bytes"] = int(free)
            # torch.cuda.is_bf16_supported() defaults to including_emulation=True and so
            # answers True on Turing, where bf16 exists only as a software conversion with
            # no tensor-core path -- picking it would be slower than fp16 for no accuracy
            # gain. Native bf16 arrives with Ampere, so gate on the capability instead.
            info["supports_bf16"] = (major, minor) >= (8, 0)
            info["bf16_emulated"] = bool(
                not info["supports_bf16"] and _try(lambda: torch.cuda.is_bf16_supported())
            )

            # The decisive check: a wheel built without this arch raises
            # "no kernel image is available for execution on the device" at the
            # first matmul, long after we would have told the user everything is fine.
            arch_list = list(torch.cuda.get_arch_list())
            info["arch_list"] = arch_list
            want = f"sm_{major}{minor}"
            info["arch_supported"] = (not arch_list) or any(a.startswith(want) for a in arch_list)
            if not info["arch_supported"]:
                info["notes"].append(
                    f"This torch build ships {', '.join(arch_list) or 'no'} kernels but the GPU "
                    f"is {want}. Reinstall torch from a CUDA index that covers {want}."
                )
            if (major, minor) < MIN_CUDA_CAPABILITY:
                info["notes"].append(
                    f"Compute capability {major}.{minor} is below the {MIN_CUDA_CAPABILITY[0]}."
                    f"{MIN_CUDA_CAPABILITY[1]} floor for 4-bit quantisation."
                )
            if not info["supports_bf16"]:
                info["notes"].append(
                    "No native bf16 on this GPU"
                    + (" (only emulated)" if info.get("bf16_emulated") else "")
                    + "; training uses fp16 with a grad scaler."
                )
        except Exception as exc:
            info["notes"].append(f"CUDA device query failed: {exc}")

    try:
        import bitsandbytes

        info["bitsandbytes"] = bitsandbytes.__version__
    except Exception as exc:
        info["bitsandbytes_error"] = str(exc)

    info["tier"] = _tier_from(info)
    return info


def _tier_from(info: dict[str, Any]) -> str:
    if not info["cuda_available"] or info["arch_supported"] is False:
        return "cpu"
    if info["bitsandbytes"] is None:
        info["notes"].append("bitsandbytes is unavailable, so 4-bit is off; falling back to the CPU preset.")
        return "cpu"
    gib = info["vram_total_bytes"] / (1024**3)
    if gib >= 11.0:
        return "gpu_12gb"
    if gib >= 7.0:
        return "gpu_8gb"
    return "gpu_4gb"


def disk_free_bytes(path: Path | None = None) -> int:
    try:
        return shutil.disk_usage(str(path or DATA_DIR)).free
    except OSError:
        return 0


# --------------------------------------------------------------------------------------
# VRAM estimation
# --------------------------------------------------------------------------------------

_CUDA_CONTEXT_BYTES = 500 * 1024**2  # driver + cuBLAS/cuDNN workspaces, measured on WDDM


def estimate_peak_vram(model_config: dict[str, Any], tc: TrainConfig) -> dict[str, int]:
    """Rough peak-memory breakdown for a LoRA run, in bytes.

    Deliberately explicit about the LM head: for a 152k-vocab student the logits and
    their fp32 cross-entropy upcast outweigh the quantised weights, which is the
    non-obvious part of fitting one of these models into 4 GB.
    """
    hidden = int(model_config.get("hidden_size", 1024))
    layers = int(model_config.get("num_hidden_layers", 24))
    vocab = int(model_config.get("vocab_size", 32000))
    inter = int(model_config.get("intermediate_size", hidden * 4))
    heads = int(model_config.get("num_attention_heads", max(1, hidden // 64)))
    kv_heads = int(model_config.get("num_key_value_heads", heads))
    head_dim = int(model_config.get("head_dim", max(1, hidden // max(1, heads))))
    tied = bool(model_config.get("tie_word_embeddings", False))

    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    per_layer = {
        "q_proj": hidden * q_out,
        "k_proj": hidden * kv_out,
        "v_proj": hidden * kv_out,
        "o_proj": q_out * hidden,
        "gate_proj": hidden * inter,
        "up_proj": hidden * inter,
        "down_proj": inter * hidden,
    }
    linear_params = sum(per_layer.values()) * layers
    embed_params = vocab * hidden * (1 if tied else 2)

    # bitsandbytes quantises nn.Linear only; embeddings stay in the compute dtype.
    # NF4 with double quantisation lands a little over 0.5 bytes/param once the
    # absmax blocks are counted.
    weight_bytes = int(embed_params * 2)
    weight_bytes += int(linear_params * (0.55 if tc.load_in_4bit else 2))

    shapes = {
        "q_proj": (hidden, q_out),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
        "o_proj": (q_out, hidden),
        "gate_proj": (hidden, inter),
        "up_proj": (hidden, inter),
        "down_proj": (inter, hidden),
    }
    lora_params = sum(
        tc.lora_r * (fan_in + fan_out)
        for name, (fan_in, fan_out) in shapes.items()
        if name in set(tc.target_modules)
    ) * layers
    # fp32 weight + grad + Adam's two moments.
    lora_bytes = int(lora_params * 16)

    tokens = tc.batch_size * tc.max_seq_len
    # fp16 logits, their fp32 upcast for the loss, and the gradient w.r.t. logits.
    logit_bytes = int(tokens * vocab * (2 + 4 + 2))

    if tc.gradient_checkpointing:
        # One saved hidden state per layer boundary, plus one layer recomputed in full.
        act_bytes = int(layers * tokens * hidden * 2 + tokens * hidden * 24 * 2)
    else:
        act_bytes = int(layers * tokens * hidden * 20 * 2)

    total = weight_bytes + lora_bytes + logit_bytes + act_bytes + _CUDA_CONTEXT_BYTES
    return {
        "weights": weight_bytes,
        "lora_and_optimizer": lora_bytes,
        "logits_and_loss": logit_bytes,
        "activations": act_bytes,
        "cuda_context": _CUDA_CONTEXT_BYTES,
        "total": total,
        "lora_trainable_params": int(lora_params),
    }
