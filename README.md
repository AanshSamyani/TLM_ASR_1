# TTL-ASR: Test-Time Learning for Automatic Speech Recognition

Adapting ASR models (Whisper) at test time using **only unlabelled audio** — no
transcription labels required. Based on the TLM paper (Hu et al., ICML 2025).

## Method overview

| Component | What it does |
|---|---|
| **Pseudo-label CE loss** | Greedy-decode a transcript, then teacher-force the decoder on it — the CE loss is the ASR analogue of "input perplexity minimisation" |
| **Sample selection** | Skip low-perplexity (easy) samples, exponentially up-weight high-perplexity ones |
| **LoRA adaptation** | Only update low-rank adapters → prevents catastrophic forgetting |

Baselines implemented: **Tent** (entropy minimisation on LayerNorm params).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management
and [ruff](https://docs.astral.sh/ruff/) for linting/formatting.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone
git clone https://github.com/AanshSamyani/TLM_ASR_1.git
cd TLM_ASR_1

# 3. Install dependencies (creates a .venv automatically)
uv sync
```

> **Note:** The first run downloads the Whisper model (~1 GB for `whisper-small`)
> and datasets (~5 GB for LibriSpeech test sets, ~1 GB for TEDLIUM).  
> On a cluster set `HF_HOME=/path/to/large/disk` to control cache location.

## Linting & formatting

```bash
# Check for lint issues
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## Experiments

### Quick smoke test (< 10 min on 1 GPU)

```bash
uv run python run_experiment.py \
    --method ttl \
    --adapt_dataset librispeech_other \
    --eval_dataset librispeech_other \
    --max_samples 50 \
    --tag smoke
```

### Experiment 1 — TTL vs Base

Measures WER before and after TTL adaptation on three benchmarks.

```bash
bash scripts/exp1_ttl_vs_base.sh
```

### Experiment 2 — TTL vs Tent vs Base

Head-to-head with the Tent entropy-minimisation baseline.

```bash
bash scripts/exp2_ttl_vs_tent.sh
```

### Experiment 3 — Sample-selection ablation (de-risk)

Sweeps the perplexity threshold P₀ to verify the selection strategy helps.

```bash
bash scripts/exp3_sample_selection.sh
```

### Run everything

```bash
bash scripts/run_all.sh
```

## Plotting results

After experiments finish, generate comparison plots:

```bash
uv run python analysis/plot_results.py --input_dir results --output_dir results
```

This produces:

| File | Description |
|---|---|
| `results/method_comparison.png` | Grouped bar chart: WER per method per dataset |
| `results/p0_sweep.png` | Line plot: WER vs P₀ threshold |
| `results/adaptation_stats.png` | Adapted/skipped sample counts |

A summary table is printed to stdout as well.

## CLI reference

```
uv run python run_experiment.py --help
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--method` | *(required)* | `base`, `ttl`, or `tent` |
| `--model` | `openai/whisper-small` | Any HuggingFace Whisper model |
| `--adapt_dataset` | same as eval | `librispeech_clean`, `librispeech_other`, `tedlium` |
| `--eval_dataset` | *(required)* | Same choices |
| `--max_samples` | all | Cap samples for quick runs |
| `--lora_rank` | 8 | LoRA rank (TTL only) |
| `--lr` | 5e-5 | Learning rate (TTL) |
| `--tent_lr` | 1e-3 | Learning rate (Tent) |
| `--sample_selection` | off | Enable perplexity-based sample selection |
| `--p0` | e³ ≈ 20.09 | Perplexity threshold |
| `--fp16` | off | Half-precision inference |
| `--gpu` | all visible | GPU index to use (0, 1, 2, 3) |
| `--tag` | "" | Appended to output filename |

## Datasets

| Name | Domain shift | Samples |
|---|---|---|
| `librispeech_clean` | In-domain (baseline) | 2,620 |
| `librispeech_other` | Harder speakers / acoustics | 2,939 |
| `tedlium` | TED talks (different vocab/style) | ~1,155 |

## Project structure

```
├── pyproject.toml             # project metadata, deps, ruff config
├── run_experiment.py          # single entry point
├── src/
│   ├── models.py              # Whisper + LoRA loading
│   ├── ttl.py                 # TTL adaptation loop
│   ├── tent.py                # Tent baseline
│   ├── sample_selection.py    # perplexity-based weighting
│   ├── data.py                # dataset loading
│   └── eval_utils.py          # WER evaluation
├── scripts/                   # per-experiment shell scripts
├── analysis/
│   └── plot_results.py        # generate comparison plots
└── results/                   # JSON outputs + PNGs
```
