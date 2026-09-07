<div align="center">

# 🎯 Ustad

**Local-First LLM Distillation Studio**

*Transform large language models into specialized, lightweight models you own and control*

![Status](https://img.shields.io/badge/Status-✅%20Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)
![GPU](https://img.shields.io/badge/GPU-Optional-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**[Features](#features) • [Quick Start](#quick-start) • [Usage Guide](#-usage-guide) • [Architecture](#-architecture) • [API Reference](#-api-reference)**

</div>

---

## 🚀 What is Ustad?

Ustad is a **professional-grade desktop application** for **knowledge distillation** — the process of transferring expertise from a large "teacher" model into a compact "student" model through fine-tuning.

### 💡 Key Innovation

Unlike traditional distillation that requires access to model internals, Ustad uses **response-based distillation**, making it work across any model family:

```
🌱 Seed Prompts → 🧠 Teacher Generates → 📚 Dataset → 🎓 Student Fine-tunes → ✨ Production Model
```

**Why this matters:** You can now specialize any model for your exact use case without expensive compute or complex infrastructure.

### 🏆 Why Ustad?

- 🔒 **100% Local** — No API keys, no cloud, no telemetry
- 🎯 **Specialized Models** — Create domain-specific models
- 💾 **Resource Efficient** — Runs on 4GB GPU (consumer hardware)
- 🔄 **Cross-Family** — Train TinyLlama from Qwen, or any combination
- 📊 **Real-Time Visualization** — Live gradient flow and training metrics
- 🎨 **Premium Interface** — Futuristic dark theme with drag-and-drop support

---

## ⚡ Features

### 🎓 Intelligent Dataset Generation
- **Self-Instruct Pipeline** — Automatically expands seeds into diverse training data
- **Quality Filtering** — Near-duplicate detection, refusal filtering, length validation
- **Human-in-the-Loop** — Review and curate before training

### 🔬 Advanced Fine-Tuning
- **QLoRA/LoRA** — Memory-efficient 4-bit quantization
- **Response-Only Loss** — Train only on teacher outputs
- **Token-Weighted Accumulation** — Proper handling of variable-length responses
- **Per-Layer Telemetry** — Real-time gradient norm tracking

### 📈 Professional Visualization
- **Live Dashboard** — Step-by-step loss curves, learning rate, throughput
- **Layer Gradient Graph** — 28-layer visual with real-time gradients
- **Resource Monitoring** — VRAM usage and training ETA
- **Dataset Preview** — Paginated view with row-level deletion

### 🎯 Hardware Adaptive
- **Automatic Presets** — GPU tier detection with optimized hyperparameters
- **VRAM Estimation** — Pre-flight checks before training
- **CPU Fallback** — Full functionality without GPU
- **Smart Resource Management** — Automatic Ollama teacher eviction
- **Hardware Advisor** — Auto-detects GPU/CPU and recommends which models fit your device with fit badges (Recommended / Tight fit / Too large / CPU only)

---

## 🎯 Hardware Advisor

Ustad automatically detects your GPU/CPU hardware and recommends which models you can train on your device — no manual guessing required.

### How It Works

1. **Auto-detection** — On startup, scans GPU name, VRAM, CPU threads, and architecture
2. **VRAM estimation** — For each of 17 catalog models, estimates memory using a param-count heuristic (4-bit QLoRA weights, LoRA adapters, logit overhead, activations, CUDA context)
3. **Fit classification** — Each model gets a badge:
   - ✓ **Recommended** — Uses <60% VRAM, comfortable headroom
   - ⚠ **Tight fit** — Uses <90% VRAM, works but tight
   - ✗ **Too large** — Exceeds available VRAM, will OOM
   - 🔄 **CPU only** — No GPU available or model too large
4. **Best fit highlight** — The largest model that fits is highlighted as the best choice
5. **Click to select** — Click any recommended model to instantly select it in the Student dropdown

### Example Output

| Your GPU | Best Fit | Why |
|----------|----------|-----|
| GTX 1650 (4GB) | Qwen3 0.6B (~2.68 GB) | Tight fit, largest that works |
| RTX 3060 (12GB) | 1-2B models | Recommended range |
| RTX 4090 (24GB) | 7B+ models | Plenty of headroom |
| CPU only | GPT-Neo 125M | CPU fallback, smallest model |

> **Tip:** The advisor uses no model downloads — it's pure math based on parameter counts and your hardware specs.

---

## 🚀 Quick Start

### Option 1: Portable Installer (Recommended)

**Perfect for end users — no Python knowledge required!**

1. **Prerequisites:**
   - Python 3.10-3.12 ([Download](https://www.python.org/downloads/))
   - Ollama ([Download](https://ollama.ai))
   - NVIDIA GPU (optional, 4GB+ for QLoRA)

2. **Run Installer:**
   ```batch
   D:\Ustad\portable_installer.bat
   ```

3. **Wait** for automatic setup (5-10 minutes)

4. **Launch:**
   ```batch
   D:\Ustad-Portable\start_ustad.bat
   ```

5. **Open Browser:** http://127.0.0.1:8177

### Option 2: From Source (Developer Mode)

**For developers and customization:**

```bash
git clone https://github.com/MadB0i/Ustad.git
cd Ustad

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Launch
python -m backend.server
```

### Option 3: Docker Container

**For containerized deployment:**

```bash
# Build image
docker build -t ustad:latest -f scripts/Dockerfile .

# Run with GPU support
docker run --gpus all -p 8177:8177 -v $(pwd)/data:/app/data ustad:latest
```

---

## 🎯 Usage Guide

### 1️⃣ Select Teacher Model

Choose a pulled Ollama model from the **Teacher** panel:

- ✅ **Fits VRAM** — Model will load on your GPU
- 🔄 **Resident** — Already loaded in Ollama
- 💭 **Thinking Model** — Has `<think>` blocks (auto-stripped)

**Recommended Teachers:**
| Model | VRAM | Best For |
|-------|------|----------|
| `qwen2.5:3b` | ~2GB | Balanced quality/speed |
| `llama3.2:3b` | ~2GB | Strong instruction following |
| `qwen2.5:7b` | ~4GB | Higher quality (needs 8GB+) |

### 2️⃣ Define Your Skill

**Skill Description** (1-2 sentences):
```
Explain Python error messages: state what the error means,
the most likely cause, and the concrete fix.
```

**Seed Prompts** (4-8 high-quality examples):
```
What does "TypeError: unhashable type: 'list'" mean?
Explain "IndentationError: unexpected indent".
Why do I get "ValueError: too many values to unpack"?
```

**Target Pairs:** 40-100 (quality > quantity)

Click **Build dataset** and watch real-time generation.

### 3️⃣ Review Dataset

Open the **Dataset** drawer:
- 📖 Page through generated Q&A pairs
- 🗑️ Delete low-quality or malformed responses
- ✅ Verify answers match your intended skill

> ⚠️ **Critical:** Bad training data teaches bad behavior. Always review!

### 4️⃣ Train Student Model

**Select Base Model:**
- `Qwen/Qwen3-0.6B` (default, 28 layers)
- `Qwen/Qwen2.5-0.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

**Custom Models:**
- Switch to **Custom Path** tab
- Enter local path or HuggingFace repo ID
- Real-time validation ensures it's a valid causal-LM

**Check VRAM Estimate** before starting.

**Adjust Hyperparameters:**
- **Epochs:** 2-4 for most tasks
- **LoRA Rank:** 8-32 (higher = more capacity)
- **Sequence Length:** 512 default
- **4-bit QLoRA:** Keep enabled for GPU efficiency

Click **Start training** and monitor:
- 📉 Live loss curves (train + eval)
- 🌐 Per-layer gradient flow visualization
- 📊 Real-time metrics (tokens/sec, VRAM, ETA)

### 5️⃣ Export & Deploy

Navigate to **Export** tab:
1. Select training run
2. Choose name
3. ✅ Merge into full model
4. Click **Export**

**Deploy to Ollama:**
```bash
cd data/exports/your-export-name
ollama create my-specialized-model -f Modelfile
ollama run my-specialized-model
```

---

## 🖥️ Creating Standalone Executable

### PyInstaller Build (Windows)

**For developers who want a single .exe:**

```powershell
# Install PyInstaller
pip install pyinstaller

# Build (large, ~2-4GB)
pyinstaller --name Ustad --onefile --console main.py

# Result: dist\Ustad.exe
```

**⚠️ Limitations:**
- Executable is very large (~2-4GB due to PyTorch)
- Still requires Ollama installed separately
- GPU drivers must be installed for GPU support

**📦 Better Alternative:** Use the **Portable Installer** (Option 1 above) for 99% of use cases.

---

## 🎨 What the Visualization Shows

Every visual element is driven by **real backend telemetry** over Server-Sent Events:

| Visual Element | Data Source |
|----------------|-------------|
| 🟡 **Amber pulse on teacher** | Model is actively generating tokens |
| 📝 **Streaming text** | Raw tokens from Ollama `/api/chat` stream |
| 📦 **Light packets** | Dataset pair written or optimizer step |
| 🌈 **Layer node brightness** | **Real LoRA gradient norm** for that layer |
| 📉 **Loss sparkline** | Per-step training loss with EMA smoothing |
| 💎 **Eval diamonds** | Held-out evaluation loss at checkpoints |
| 🔢 **Numeric readouts** | Direct from training loop |

> **Key Point:** No synthetic data. When idle, nothing animates. The visualization is a **debugger**, not a screensaver.

---

## 📋 Command-Line Interface

The UI is optional. Everything accessible via CLI:

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

## 🏗️ Architecture

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
│   ├── config.py           # Hardware detection, presets, VRAM estimator, model advisor
│   ├── events.py           # Thread-safe event bus with SSE replay
│   ├── ollama_client.py    # Async Ollama API wrapper
│   ├── dataset.py          # Self-instruct + teacher response generation
│   ├── distill.py          # QLoRA training loop with telemetry
│   ├── jobs.py             # Single-slot job manager
│   ├── export.py           # CPU fp16 merge + Modelfile generation
│   └── server.py           # FastAPI routes + SSE endpoint
├── frontend/
│   ├── index.html          # Single-page application
│   ├── app.js              # UI logic + SSE client + visualization
│   └── style.css           # Premium dark theme
├── scripts/
│   ├── quickstart.ps1      # Windows automated setup
│   ├── quickstart.sh       # Linux/macOS automated setup
│   ├── portable_installer.bat  # Standalone installer
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

## 🔧 Technical Deep Dive

### Response-Based Distillation

**How it works:**
1. Teacher generates text responses
2. Student learns to predict those responses token-by-token
3. No architectural coupling required

**Advantages:**
- ✅ Cross-family distillation (Qwen → TinyLlama)
- ✅ Works with any API/inference engine
- ✅ No teacher model weights needed
- ✅ Naturally preserves output style and format

### Response-Only Loss Masking

Training loss is computed **only on the teacher's response**, not the prompt:

```python
# Template user turn alone to find prefix length
prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True)
# Mask prompt tokens to -100
labels[:prefix_length] = -100
```

### Token-Weighted Gradient Accumulation

Standard accumulation treats all micro-batches equally, which over-weights short responses. Ustad weights by **label token count**:

```python
# Weighted by actual label tokens
label_tokens = (labels != -100).sum()
weight = label_tokens / total_tokens_in_window
(loss * weight).backward()
```

### Memory Optimization

**For Qwen3-0.6B (152k vocab), logits dominate VRAM:**

| Component | Memory (seq=512, batch=1) |
|-----------|---------------------------|
| 4-bit weights (NF4) | ~546 MB |
| LoRA adapters + Adam | ~160 MB |
| fp16 logits + fp32 CE + gradients | ~800 MB |
| CUDA context | ~500 MB |
| **Total peak** | **~2.1 GB** (fits 4GB) |

---

## 🛠️ Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **"Ollama unreachable"** | Start daemon: `ollama serve` |
| **"No chat template"** | Model is completion-only. Pull an instruct variant. |
| **Out of GPU memory** | Lower seq length (512→256), then LoRA rank (16→8). Check `ollama ps`. |
| **Training is slow on CPU** | Expected. Use `--max-steps` to limit run time. |
| **Loss is flat** | Usually dataset quality. Check for truncated/refused answers. |
| **Train loss ↓ but eval loss ↑** | Overfitting. Use fewer epochs or Max steps. |

### Hardware-Specific Notes

**GTX 1650 (4GB, sm_75):**
- ✅ Use `cu126` PyTorch wheels
- ✅ Enable 4-bit QLoRA
- ✅ Keep seq_len ≤ 512
- ⚠️ Expect ~50-80 tokens/sec

**RTX 3060 (12GB, sm_86):**
- ✅ Can use bf16
- ✅ Increase seq_len to 1024
- ✅ Increase batch_size to 2-4
- ✅ Expect ~200-300 tokens/sec

**CPU Only (16GB RAM):**
- ✅ Use CPU preset
- ⚠️ Expect ~5-10 tokens/sec
- ✅ Fully functional, just slower

---

## 📡 API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system` | GET | Hardware info, presets, Ollama status, model recommendations |
| `/api/teachers` | GET | List Ollama models with template detection |
| `/api/teachers/unload` | POST | Evict resident teacher from GPU |
| `/api/students/download` | POST | Download student model from HuggingFace |
| `/api/students/browse` | POST | Validate custom model path |
| `/api/datasets` | GET | List generated datasets |
| `/api/datasets/{name}` | GET | Paginated dataset view |
| `/api/datasets/{name}` | DELETE | Delete dataset |
| `/api/datasets/{name}/rows/{id}` | DELETE | Delete single row |
| `/api/dataset/build` | POST | Start dataset generation job |
| `/api/train/start` | POST | Start training job |
| `/api/train/stop` | POST | Stop current job |
| `/api/estimate` | POST | Estimate VRAM for config |
| `/api/runs` | GET | List training runs |
| `/api/runs/{id}` | GET | Run details with metrics |
| `/api/export` | POST | Start export job |
| `/api/exports` | GET | List exports |
| `/api/events` | GET | Server-Sent Events stream |

### SSE Event Types

```typescript
type Event =
  | { kind: "system", message: string }
  | { kind: "dataset.phase", phase: string, count: number }
  | { kind: "dataset.pair", id: string, response: string }
  | { kind: "train.step", step: number, loss: number, lr: number }
  | { kind: "train.layer_grads", step: number, norms: number[] }
  | { kind: "train.eval", step: number, eval_loss: number }
  | { kind: "train.done", run_id: string }
  | { kind: "job.state", phase: "idle" | "dataset" | "training" }
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- 🔧 Additional student model presets
- 📊 Alternative visualization modes
- 🧪 Evaluation harness integration
- 🌐 Multi-language UI
- 📦 Docker/container improvements

**Development Setup:**
```bash
git clone https://github.com/MadB0i/Ustad.git
cd Ustad
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

**Built with:**
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Transformers](https://github.com/huggingface/transformers) — Model loading
- [PEFT](https://github.com/huggingface/peft) — LoRA adapters
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization
- [FastAPI](https://fastapi.tiangolo.com/) — Web framework
- [Ollama](https://ollama.ai/) — Local LLM serving

**Inspired by:**
- [Self-Instruct](https://arxiv.org/abs/2212.10560) — Automated instruction generation
- [QLoRA](https://arxiv.org/abs/2305.14314) — Efficient fine-tuning
- [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301) — Task-specific distillation

---

<div align="center">

### 🎉 Ready to specialize your own models?

**[Get Started](#-quick-start) • [Report Bug](https://github.com/MadB0i/Ustad/issues) • [Request Feature](https://github.com/MadB0i/Ustad/issues)**

**Built with ❤️ for the local LLM community**

</div>