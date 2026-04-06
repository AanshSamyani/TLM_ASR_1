# TTL-ASR: Test-Time Learning for Automatic Speech Recognition

Adapting ASR models (Whisper) at test time using **only unlabelled audio** — no
transcription labels required. Based on the TLM paper (Hu et al., ICML 2025).

## Method overview

| Component | What it does |
|---|---|
| **Entropy minimisation** | Minimise decoder output entropy — the ASR analogue of TLM's input perplexity |
| **LoRA adaptation** | Update low-rank adapters in attention layers (884K params) |
| **Sample selection** | Perplexity-based weighting with threshold P0 |

Baselines implemented: **Tent** (entropy on LayerNorm params), **SUTA** (entropy + MCC, per-utterance reset, multi-step).

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

# 4. Create directories for logs and plots
mkdir -p logs plots
```

> **Note:** The first run downloads the Whisper model (~1 GB for `whisper-small`)
> and datasets (~5 GB for LibriSpeech test sets, ~1 GB for TEDLIUM).
> On a cluster set `HF_HOME=/path/to/large/disk` to control cache location.

## Running experiments

All experiments are designed to be run with `nohup` on a remote server. Logs are
saved to `logs/`, result JSONs to `results/`, and plots to `plots/`.

### Phase 1: Core experiments (exp1–exp4)

Runs all clean baselines, noise robustness (LibriSpeech-C), and sample selection ablation.

```bash
# Run everything in Phase 1 (~3 days GPU on A100)
mkdir -p logs
nohup bash scripts/run_phase1.sh --gpu 0 --batch_size 32 > logs/phase1.log 2>&1 &

# Monitor progress
tail -f logs/phase1.log
```

Or run individual experiments:

```bash
# Exp 1: TTL vs Base (3 datasets)
nohup bash scripts/exp1_ttl_vs_base.sh --gpu 0 --batch_size 32 > logs/exp1.log 2>&1 &

# Exp 2: TTL vs Tent + LR sweep
nohup bash scripts/exp2_ttl_vs_tent.sh --gpu 0 --batch_size 32 > logs/exp2.log 2>&1 &

# Exp 3: Sample selection ablation
nohup bash scripts/exp3_sample_selection.sh --gpu 0 --batch_size 32 > logs/exp3.log 2>&1 &

# Exp 4: Noise robustness (Gaussian + Babble, 5 SNR levels, 4 methods, + Common Voice)
nohup bash scripts/exp4_librispeech_c.sh --gpu 0 --batch_size 32 > logs/exp4.log 2>&1 &
```

### Phase 2: Ablations + model scaling (exp5–exp6)

```bash
# Run everything in Phase 2 (~2 days GPU)
nohup bash scripts/run_phase2.sh --gpu 0 --batch_size 32 > logs/phase2.log 2>&1 &
```

Or individually:

```bash
# Exp 5: Model scaling (whisper-tiny through whisper-large-v3)
nohup bash scripts/exp5_model_scaling.sh --gpu 0 --batch_size 32 > logs/exp5.log 2>&1 &

# Exp 6: LoRA ablations (rank sweep, LR sweep, placement)
nohup bash scripts/exp6_ablations.sh --gpu 0 --batch_size 32 > logs/exp6.log 2>&1 &
```

### Running on multiple GPUs in parallel

To maximise throughput, run different experiments on different GPUs:

```bash
mkdir -p logs
nohup bash scripts/exp4_librispeech_c.sh --gpu 0 --batch_size 32 > logs/exp4.log 2>&1 &
nohup bash scripts/exp5_model_scaling.sh --gpu 1 --batch_size 32 > logs/exp5.log 2>&1 &
nohup bash scripts/exp6_ablations.sh --gpu 2 --batch_size 32 > logs/exp6.log 2>&1 &
```

### Quick smoke test (< 10 min)

```bash
uv run python run_experiment.py \
    --method ttl --ppl_method entropy \
    --adapt_dataset librispeech_other \
    --eval_dataset librispeech_other \
    --max_samples 50 --tag smoke --gpu 0
```

## Plotting results

Plots are auto-generated at the end of each experiment script. To regenerate manually:

```bash
uv run python analysis/plot_results.py --input_dir results --output_dir plots
```

| File | Description | Experiment |
|---|---|---|
| `plots/method_comparison.png` | Grouped bar chart: WER per method per dataset | exp1–4 |
| `plots/p0_sweep.png` | Line plot: WER vs P0 threshold | exp3 |
| `plots/adaptation_stats.png` | Adapted/skipped sample counts | all |
| `plots/noise_severity.png` | WER vs SNR for Gaussian/Babble noise | exp4 |
| `plots/model_scaling.png` | WER across Whisper sizes (tiny→large-v3) | exp5 |
| `plots/lora_rank_sweep.png` | WER vs LoRA rank | exp6 |
| `plots/lora_lr_sweep.png` | WER vs learning rate | exp6 |
| `plots/lora_placement.png` | WER by LoRA placement (enc/dec/both) | exp6 |
| `plots/entropy_histogram.png` | Per-sample decoder entropy distribution | exp1–4 |

## CLI reference

```
uv run python run_experiment.py --help
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--method` | *(required)* | `base`, `ttl`, `tent`, or `suta` |
| `--model` | `openai/whisper-small` | Any HuggingFace Whisper model |
| `--adapt_dataset` | same as eval | `librispeech_clean`, `librispeech_other`, `tedlium`, `common_voice_en` |
| `--eval_dataset` | *(required)* | Same choices |
| `--max_samples` | all | Cap samples for quick runs |
| `--gpu` | all visible | GPU index (0, 1, 2, 3) |
| `--fp16` | off | Half-precision (use for medium/large models) |
| `--tag` | "" | Appended to output filename |

### TTL-specific flags

| Flag | Default | Description |
|---|---|---|
| `--ppl_method` | `entropy` | Loss mode: `ce`, `entropy`, or `gen` |
| `--lora_rank` | 8 | LoRA rank |
| `--lora_targets` | `q_proj v_proj` | Modules to apply LoRA to |
| `--lora_placement` | `both` | `encoder`, `decoder`, or `both` |
| `--lr` | 5e-5 | Learning rate (use 1e-3 for best results) |
| `--sample_selection` | off | Enable perplexity-based sample selection |
| `--p0` | e^3 | Perplexity threshold |

### Tent flags

| Flag | Default | Description |
|---|---|---|
| `--tent_lr` | 1e-3 | Learning rate |

### SUTA flags

| Flag | Default | Description |
|---|---|---|
| `--suta_lr` | 1e-3 | Learning rate |
| `--suta_steps` | 10 | Gradient steps per utterance |
| `--suta_alpha` | 0.5 | Entropy weight (1-alpha for MCC). 1.0 = entropy-only |
| `--suta_temperature` | 2.0 | Logit temperature smoothing |

### Noise injection flags

| Flag | Default | Description |
|---|---|---|
| `--noise_type` | `none` | `gaussian` or `babble` |
| `--noise_snr` | 10.0 | Signal-to-noise ratio in dB |

## Datasets

| Name | Domain | Samples |
|---|---|---|
| `librispeech_clean` | Read speech (baseline) | 2,620 |
| `librispeech_other` | Harder speakers/acoustics | 2,939 |
| `tedlium` | TED talks (domain shift) | ~1,155 |
| `common_voice_en` | Crowdsourced (diverse accents) | ~16,000 |

## Methods

| Method | Params updated | Key difference |
|---|---|---|
| **Base** | None | No adaptation (frozen model) |
| **Tent** | LayerNorm (~95K) | Single-pass entropy minimisation |
| **SUTA** | LayerNorm (~95K) | Per-utterance reset, multi-step, entropy + MCC |
| **TTL** | LoRA (~884K) | Entropy minimisation through attention LoRA adapters |

## Project structure

```
├── pyproject.toml                 # deps, ruff config
├── run_experiment.py              # single entry point for all methods
├── src/
│   ├── models.py                  # Whisper + LoRA loading
│   ├── ttl.py                     # TTL adaptation (entropy/CE/gen)
│   ├── tent.py                    # Tent baseline
│   ├── suta.py                    # SUTA baseline (entropy + MCC)
│   ├── noise.py                   # Noise injection (Gaussian, Babble)
│   ├── sample_selection.py        # Perplexity-based weighting
│   ├── data.py                    # Dataset loading + collators
│   └── eval_utils.py              # WER evaluation
├── scripts/
│   ├── exp1_ttl_vs_base.sh        # TTL vs base across datasets
│   ├── exp2_ttl_vs_tent.sh        # TTL vs Tent + LR sweep
│   ├── exp3_sample_selection.sh   # P0 threshold ablation
│   ├── exp4_librispeech_c.sh      # Noise robustness (Gaussian + Babble)
│   ├── exp5_model_scaling.sh      # Whisper size sweep
│   ├── exp6_ablations.sh          # LoRA rank, LR, placement
│   ├── run_phase1.sh              # Run exp1–4 sequentially
│   └── run_phase2.sh              # Run exp5–6 sequentially
├── analysis/
│   └── plot_results.py            # Generate all comparison plots
├── results/                       # JSON outputs
├── plots/                         # Generated PNG plots
└── logs/                          # nohup log files
```

## Linting & formatting

```bash
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .
```
