# Ustad

<div align="center">

**A Local-First LLM Distillation Studio**

*Transform large language models into specialized, lightweight models you own and control*

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU: Optional](https://img.shields.io/badge/GPU-Optional-green.svg)](https://pytorch.org/)

[Features](#features) • [Quick Start](#quick-start) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation)

</div>

---

## Overview

Ustad is a professional-grade desktop application for **knowledge distillation** — the process of transferring expertise from a large "teacher" model into a compact "student" model through fine-tuning. Unlike traditional distillation that requires access to model internals, Ustad uses **response-based distillation**, making it work across any model family.

**Key Principle:** Your locally-running Ollama model becomes the teacher. You describe a skill, provide example prompts, and Ustad handles the rest — generating training data, fine-tuning a student model, and delivering a production-ready export.

### Why Ustad?

- 🔒 **100% Local** — No API keys, no cloud inference, no telemetry after initial model downloads
- 🎯 **Specialized Models** — Create domain-specific models optimized for your exact use case
- 💾 **Resource Efficient** — Runs on consumer hardware (4GB GPU sufficient with QLoRA)
- 🔄 **Cross-Family Distillation** — Train TinyLlama students from Qwen teachers, or any combination
- 📊 **Real-Time Visualization** — Live training metrics and gradient flow visualization
- 🚀 **Production Ready** — Export directly to Ollama or GGUF format

---

## Features

### 🎓 Intelligent Dataset Generation
- **Self-Instruct Pipeline**: Automatically expands seed prompts into diverse training data
- **Quality Filtering**: Near-duplicate detection, refusal filtering, and length validation
- **Human-in-the-Loop**: Review and curate generated pairs before training

### 🔬 Advanced Fine-Tuning
- **QLoRA/LoRA Support**: Memory-efficient 4-bit quantization for consumer GPUs
- **Response-Only Loss**: Train only on teacher outputs, not prompts
- **Token-Weighted Accumulation**: Proper handling of variable-length responses
- **Per-Layer Telemetry**: Real-time gradient norm tracking for each transformer layer

### 📈 Professional Visualization
- **Live Training Dashboard**: Step-by-step loss curves, learning rate, and throughput
- **Layer Gradient Graph**: 28-layer visual representation with real-time gradient norms
- **Resource Monitoring**: VRAM usage and training ETA
- **Dataset Preview**: Paginated view with row-level deletion

### 🎯 Hardware Adaptive
- **Automatic Preset Selection**: GPU tier detection with optimized hyperparameters
- **VRAM Estimation**: Pre-flight checks before training starts
- **CPU Fallback**: Full functionality without GPU (slower, but functional)
- **Smart Resource Management**: Automatic Ollama teacher eviction before training

---

## Quick Start

### Prerequisites

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.10, 3.11, or 3.12 (PyTorch has no 3.13+ wheels yet) |
| **Ollama** | Installed with at least one instruction-tuned model pulled |
| **GPU** | Optional. 4GB+ NVIDIA (compute capability ≥ 6.0) for 4-bit QLoRA |
| **RAM** | 8GB+ (16GB recommended for merging exports) |
| **Disk** | ~5GB per teacher + student + merged export |

### Installation

**Windows (PowerShell):**
```powershell
git clone https://github.com/yourusername/Ustad.git
cd Ustad
.\scripts\quickstart.ps1
```

**Linux/macOS (Bash):**
```bash
git clone https://github.com/yourusername/Ustad.git
cd Ustad
./scripts/quickstart.sh
```

The quickstart script will:
1. ✅ Verify Python 3.10-3.12
2. 📦 Create virtual environment
3. 🔧 Install PyTorch (CUDA 12.6 or CPU)
4. 📚 Install dependencies
5. 🚀 Start Ollama daemon (if needed)
6. 🌐 Launch app at http://127.0.0.1:8177

### Manual Installation

If you prefer manual control:

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/macOS)
source .venv/bin/activate

# Install PyTorch (choose one):
# For NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu126
# For CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Start server
python -m backend.server
```

> **Note:** PyTorch is deliberately **not** in `requirements.txt` because the correct wheel depends on your hardware. Pinning a CUDA build would break CPU-only installs.

---

## Usage

### 1️⃣ Select a Teacher Model

Choose a pulled Ollama model from the **Teacher** panel. Status badges indicate:
- ✅ **Fits VRAM**: Model will load on your GPU
- 🔄 **Resident**: Already loaded in Ollama
- 💭 **Thinking Model**: Has `<think>` blocks (auto-stripped)

**Recommended Teachers:**
- `qwen2.5:3b` — Balanced quality/speed, fits 4GB
- `llama3.2:3b` — Strong instruction following
- `qwen2.5:7b` — Higher quality (needs 8GB+ VRAM)

### 2️⃣ Define Your Skill

**Skill Description** (1-2 sentences):
```
Explain Python error messages: state what the error means, 
the most likely cause, and the concrete fix.
```

**Seed Prompts** (4-8 high-quality examples):
```
What does "TypeError: unhashable type: 'list'" mean and how do I fix it?
Explain "IndentationError: unexpected indent" and how to resolve it.
Why do I get "ValueError: too many values to unpack"?
```

**Target Pairs**: 40-100 (more isn't always better; quality > quantity)

Click **Build dataset** and watch real-time generation.

### 3️⃣ Review Dataset

Open the **Dataset** drawer at the bottom:
- 📖 Page through generated Q&A pairs
- 🗑️ Delete low-quality or malformed responses
- ✅ Verify answers match your intended skill

> **Critical:** Bad training data teaches bad behavior. A refused or truncated answer trains the student to refuse and truncate.

### 4️⃣ Train Student Model

**Select Base Model:**
- `Qwen/Qwen3-0.6B` (default, 28 layers, 152k vocab)
- `Qwen/Qwen2.5-0.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

**Check VRAM Estimate:** Shows peak memory before training starts.

**Adjust Hyperparameters** (optional):
- **Epochs**: 2-4 for most tasks
- **LoRA Rank**: 8-32 (higher = more capacity, more memory)
- **Sequence Length**: 512 default (lower if OOM)
- **4-bit QLoRA**: Keep enabled for GPU efficiency

Click **Start training** and monitor:
- 📉 Live loss curves (train + eval)
- 🌐 Per-layer gradient flow visualization
- 📊 Real-time metrics (tokens/sec, VRAM, ETA)

### 5️⃣ Export & Deploy

Navigate to the **Export** tab:
1. Select your training run
2. Choose a name
3. ✅ **Merge into full model** (recommended)
4. Click **Export**

**Deploy to Ollama:**
```bash
cd data/exports/your-export-name
ollama create my-specialized-model -f Modelfile
ollama run my-specialized-model
```

**Quantize to GGUF** (optional, for smaller file size):
```bash
# Commands are in the export's README.md
python llama.cpp/convert_hf_to_gguf.py merged/
llama-quantize merged.gguf q4_k_m.gguf Q4_K_M
```

---

## Creating a Standalone Executable

You can package Ustad as a standalone `.exe` (Windows) or binary (Linux/macOS) that doesn't require Python installation.

### Using PyInstaller

**Install PyInstaller:**
```bash
pip install pyinstaller
```

**Create Executable:**
```powershell
# Windows
pyinstaller --name Ustad `
  --onefile `
  --add-data "frontend;frontend" `
  --add-data "backend;backend" `
  --hidden-import uvicorn.logging `
  --hidden-import uvicorn.loops.auto `
  --hidden-import uvicorn.protocols.http.auto `
  --icon=frontend/icon.ico `
  backend/server.py
```

```bash
# Linux/macOS
pyinstaller --name Ustad \
  --onefile \
  --add-data "frontend:frontend" \
  --add-data "backend:backend" \
  --hidden-import uvicorn.logging \
  --hidden-import uvicorn.loops.auto \
  --hidden-import uvicorn.protocols.http.auto \
  backend/server.py
```

The executable will be in `dist/Ustad.exe` (Windows) or `dist/Ustad` (Linux/macOS).

**Limitations of Executable Build:**
- ⚠️ PyTorch/transformers are large (~2-4GB executable)
- ⚠️ First run is slower due to extraction
- ⚠️ GPU drivers must still be installed separately
- ⚠️ Ollama must still be installed and running

### Alternative: Docker Container

For cross-platform distribution, consider a Docker image instead:

```dockerfile
# See scripts/Dockerfile for full implementation
FROM python:3.12-slim
# ... (full Dockerfile provided in repository)
```

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Browser)                       │
│  Vanilla JS/HTML/CSS • Real-time SSE • Canvas Visualization    │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/REST + Server-Sent Events
┌────────────────────────▼────────────────────────────────────────┐
│                    FastAPI Backend (Python)                      │
│  • Job Manager (single GPU consumer)                            │
│  • Event Bus (thread-safe SSE fan-out)                          │
│  • Ollama Client (async httpx)                                  │
└─────────┬───────────────────────────────┬───────────────────────┘
          │                               │
┌─────────▼──────────┐         ┌─────────▼──────────┐
│  Ollama (Teacher)  │         │ PyTorch + PEFT     │
│  • Data Generation │         │ • QLoRA Training   │
│  • Local Inference │         │ • Adapter Merge    │
└────────────────────┘         └────────────────────┘
```

### Project Structure

```
Ustad/
├── backend/
│   ├── config.py           # Hardware detection, presets, VRAM estimator
│   ├── events.py           # Thread-safe event bus with SSE replay
│   ├── ollama_client.py    # Async Ollama API wrapper
│   ├── dataset.py          # Self-instruct + teacher response generation
│   ├── distill.py          # QLoRA training loop with telemetry
│   ├── jobs.py             # Single-slot job manager
│   ├── export.py           # CPU fp16 merge + Modelfile generation
│   └── server.py           # FastAPI routes + SSE endpoint
├── frontend/
│   ├── index.html          # Single-page application
│   ├── app.js              # UI logic + SSE client
│   └── style.css           # Premium dark theme
├── scripts/
│   ├── quickstart.ps1      # Windows automated setup
│   ├── quickstart.sh       # Linux/macOS automated setup
│   └── smoke_test.py       # CLI verification suite
├── data/                   # Runtime data (gitignored)
│   ├── hf-cache/           # Hugging Face model cache
│   ├── datasets/           # Generated JSONL datasets
│   ├── runs/               # Training run outputs
│   └── exports/            # Merged models + Modelfiles
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## Visualization Explained

Every visual element is driven by real backend telemetry over Server-Sent Events (SSE):

| Visual Element | Data Source |
|----------------|-------------|
| 🟡 Amber pulse on teacher card | Model is actively generating tokens |
| 📝 Streaming text | Raw tokens from Ollama `/api/chat` stream |
| 📦 Light packets | Dataset pair written or optimizer step completed |
| 🌈 Layer node brightness | **Real LoRA gradient norm** for that layer |
| 📉 Loss sparkline | Per-step training loss with EMA smoothing |
| 💎 Eval diamonds | Held-out evaluation loss at checkpoints |
| 🔢 Numeric readouts | Direct from training loop (step, loss, lr, grad norm, tok/s, VRAM, ETA) |

**No synthetic data.** When idle, nothing animates. The visualization is a **debugger**, not a screensaver.

Gradient norms are captured **after** `scaler.unscale_()` and **before** clipping, representing true gradient magnitudes.

`prefers-reduced-motion` disables animations while preserving all numeric data.

---

## Command-Line Interface

The UI is optional. All functionality is accessible via `smoke_test.py`:

```bash
# Hardware preflight check
python scripts/smoke_test.py --check

# Generate dataset (24 pairs)
python scripts/smoke_test.py --build --n 24

# Train for 30 steps
python scripts/smoke_test.py --train --max-steps 30

# Merge and export
python scripts/smoke_test.py --export

# Full end-to-end pipeline
python scripts/smoke_test.py --all
```

**`--check` verifies:**
- ✅ Compute capability (e.g., sm_75 for GTX 1650)
- ✅ `torch.cuda.get_arch_list()` includes your GPU
- ✅ Free VRAM and RAM
- ✅ bitsandbytes 4-bit support
- ✅ Ollama daemon reachability
- ✅ Actual fp16 matmul + NF4 layer execution

---

## Technical Deep Dive

### Response-Based Distillation

Unlike traditional knowledge distillation (which requires access to teacher logits), Ustad uses **response-based distillation**:

1. Teacher generates text responses
2. Student learns to predict those responses token-by-token
3. No architectural coupling required

**Advantages:**
- ✅ Cross-family distillation (Qwen → TinyLlama)
- ✅ Works with any API/inference engine
- ✅ No teacher model weights needed
- ✅ Naturally preserves output style and format

### Response-Only Loss Masking

The training loss is computed **only on the teacher's response**, not the prompt:

```python
# Template user turn alone to find prefix length
prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], 
                                       add_generation_prompt=True)
# Mask prompt tokens to -100 (ignored by CrossEntropyLoss)
labels[:prefix_length] = -100
```

Verified token-by-token because BPE can merge across the prompt/response boundary.

### Token-Weighted Gradient Accumulation

Standard gradient accumulation (`loss / grad_accum`) treats all micro-batches equally, which over-weights short responses. Ustad weights by **label token count**:

```python
# Each micro-batch weighted by its share of accumulation window's tokens
label_tokens = (labels != -100).sum()
total_tokens_in_window = sum(all_label_tokens)
weight = label_tokens / total_tokens_in_window
(loss * weight).backward()
```

Critical when response lengths vary 10x (common in instruction data).

### Memory Optimization

**For Qwen3-0.6B (152k vocab), logits dominate VRAM:**

| Component | Memory (seq=512, batch=1) |
|-----------|---------------------------|
| 4-bit weights (NF4) | ~546 MB |
| LoRA adapters + Adam states | ~160 MB |
| fp16 logits + fp32 CE + gradients | ~800 MB |
| CUDA context | ~500 MB |
| **Total peak** | **~2.1 GB** (fits 4GB) |

This is why default is `seq_len=512` and `batch_size=1` with gradient accumulation, not larger batches.

### fp16 vs bf16

`torch.cuda.is_bf16_supported()` returns `True` on Turing due to software emulation, but there are **no bf16 Tensor Cores before Ampere**. Ustad checks compute capability:

```python
if compute_capability >= 8.0:  # Ampere+
    use_bf16 = True
else:  # Turing and earlier
    use_fp16_with_grad_scaler = True
```

Gradient scaling prevents underflow; skipped steps (from overflow) don't advance the LR schedule.

### Job Manager: One GPU Consumer at a Time

Ollama keeps teachers resident for 5 minutes by default. On a 4GB GPU, this would OOM during training. Solution:

```python
# Before training:
1. jobs.acquire_slot()  # Block if dataset generation running
2. ollama.unload(teacher, keep_alive=0)  # Explicit eviction
3. wait_for_vram_release()  # Poll until driver confirms
4. load_student_for_training()
```

Single-slot job manager serializes: dataset generation → training → export.

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **"Ollama unreachable"** | Start the daemon: `ollama serve` |
| **"No chat template"** | Model is completion-only, not instruction-tuned. Pull an instruct variant. |
| **Out of GPU memory** | Lower seq length (512→256), then LoRA rank (16→8), then try CPU preset. Check `ollama ps` for resident models. |
| **Training is slow on CPU** | Expected. CPU preset uses seq=256, rank=8, no gradient checkpointing. Consider `--max-steps` to limit run time. |
| **Loss is flat** | Usually dataset quality. Check for truncated/refused answers in Dataset drawer. 20 good pairs > 200 mediocre ones. |
| **Train loss ↓ but eval loss ↑** | Overfitting. You're past the useful point. This bites on small datasets: 22 rows × accum 8 = ~3 steps/epoch, so 30 steps ≈ 10 epochs. |

### Hardware-Specific Notes

**GTX 1650 (4GB, sm_75):**
- ✅ Use `cu126` PyTorch wheels (cu128+ drop Turing)
- ✅ Enable 4-bit QLoRA
- ✅ Keep seq_len ≤ 512
- ⚠️ Expect ~50-80 tokens/sec during training

**RTX 3060 (12GB, sm_86):**
- ✅ Can use bf16 (better than fp16)
- ✅ Increase seq_len to 1024
- ✅ Increase batch_size to 2-4
- ✅ Expect ~200-300 tokens/sec

**CPU Only (16GB RAM):**
- ✅ Use CPU preset (auto-selected)
- ⚠️ Expect ~5-10 tokens/sec
- ⚠️ Training will take 10-20x longer
- ✅ Fully functional, just slower

---

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system` | GET | Hardware info, presets, Ollama status |
| `/api/teachers` | GET | List Ollama models with chat template detection |
| `/api/teachers/unload` | POST | Evict resident teacher from GPU |
| `/api/students/download` | POST | Download student model from Hugging Face |
| `/api/students/browse` | POST | Validate custom model path |
| `/api/datasets` | GET | List generated datasets |
| `/api/datasets/{name}` | GET | Paginated dataset view |
| `/api/datasets/{name}` | DELETE | Delete dataset |
| `/api/datasets/{name}/rows/{id}` | DELETE | Delete single row |
| `/api/dataset/build` | POST | Start dataset generation job |
| `/api/train/start` | POST | Start training job |
| `/api/train/stop` | POST | Stop current job |
| `/api/estimate` | POST | Estimate VRAM for training config |
| `/api/runs` | GET | List training runs |
| `/api/runs/{id}` | GET | Run details with metrics |
| `/api/export` | POST | Start export job |
| `/api/exports` | GET | List exports |
| `/api/events` | GET | Server-Sent Events stream (SSE) |

### SSE Event Types

```typescript
type Event = 
  | { kind: "system", message: string }
  | { kind: "dataset.phase", phase: string, count: number }
  | { kind: "dataset.prompt", prompt: string }
  | { kind: "dataset.pair", id: string, prompt: string, response: string }
  | { kind: "dataset.done", count: number }
  | { kind: "train.start", run_id: string, student: string }
  | { kind: "train.step", step: number, loss: number, lr: number, ... }
  | { kind: "train.layer_grads", step: number, norms: number[] }
  | { kind: "train.eval", step: number, eval_loss: number }
  | { kind: "train.done", run_id: string }
  | { kind: "job.state", phase: "idle" | "dataset" | "training" | "export" }
```

---

## Contributing

Contributions are welcome! Areas of interest:

- 🔧 Additional student model presets
- 📊 Alternative visualization modes
- 🧪 Evaluation harness integration
- 🌐 Multi-language UI
- 📦 Docker/container improvements

**Development Setup:**
```bash
git clone https://github.com/yourusername/Ustad.git
cd Ustad
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt  # linting, testing
```

**Code Style:**
- Black for formatting
- Ruff for linting
- Type hints required for public functions
- Docstrings for non-obvious logic

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Citation

If you use Ustad in academic research:

```bibtex
@software{ustad2026,
  title = {Ustad: Local-First LLM Distillation Studio},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/Ustad}
}
```

---

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Transformers](https://github.com/huggingface/transformers) — Model loading and inference
- [PEFT](https://github.com/huggingface/peft) — LoRA adapters
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization
- [FastAPI](https://fastapi.tiangolo.com/) — Web framework
- [Ollama](https://ollama.ai/) — Local LLM serving

Inspired by:
- [Self-Instruct](https://arxiv.org/abs/2212.10560) — Automated instruction generation
- [QLoRA](https://arxiv.org/abs/2305.14314) — Efficient fine-tuning
- [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301) — Task-specific distillation

---

<div align="center">

**Built with ❤️ for the local LLM community**

[Report Bug](https://github.com/yourusername/Ustad/issues) • [Request Feature](https://github.com/yourusername/Ustad/issues) • [Documentation](https://github.com/yourusername/Ustad/wiki)

</div>
