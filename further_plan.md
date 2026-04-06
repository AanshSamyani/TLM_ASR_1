# Plan: Research Paper Experiment Pipeline for TTL-ASR

## Context

We adapted the TLM framework (ICML 2025) for encoder-decoder ASR. Key finding: CE-based perplexity minimization fails for ASR (PPL ~1.27, no gradient), but **entropy minimization through LoRA** works — achieving 13.77% WER on TEDLIUM vs Tent's 15.19% (9.3% relative improvement). 

To publish this, we need: more models, more datasets, proper baselines (SUTA is critical), ablations, and statistical significance testing. The paper narrative is "Bridging Test-Time Learning from LLMs to Speech."

**Target venue**: EMNLP 2026 (ARR May 25, ~7 weeks) or ICASSP 2027 (~Sep 2026).

---

## Phase 1: Core Experiments (Weeks 1-3) — MUST-HAVE

### 1a. Implement SUTA baseline (~1 day)
- **New file**: `src/suta.py`
- SUTA = entropy + minimum class confusion (MCC) loss, per-utterance reset, multi-step
- `L = alpha * H(y) + (1-alpha) * L_mcc(y)` with temperature smoothing (T=2.0)
- Operates on LayerNorm params (like Tent), resets after each utterance, K=10 steps/utterance
- Add `--method suta` + `--suta_steps`, `--suta_alpha` to `run_experiment.py`
- Reference: SUTA GitHub (`DanielLin94144/Test-time-adaptation-ASR-SUTA`)

### 1b. Implement LibriSpeech-C noise corruption (~1 day)
- **New file**: `src/noise.py` — on-the-fly noise injection in the data collator
- `add_gaussian_noise(audio, snr_db)`, `add_babble_noise(audio, snr_db)`
- 5 severity levels: SNR = {20, 15, 10, 5, 0} dB
- Add `--noise_type {none,gaussian,babble,music}` and `--noise_snr` flags to `run_experiment.py`
- Modify `src/data.py`: `NoisyWhisperCollator` wrapping existing `WhisperCollator`

### 1c. Run core experiments (~3 days GPU)
**All on whisper-small:**

| Experiment | Datasets | Methods | Runs |
|-----------|----------|---------|------|
| Clean baselines | LS-clean, LS-other, TEDLIUM | base, tent, suta, ttl | 12 |
| LS-C Gaussian | LS-other + 5 SNR levels | base, tent, suta, ttl | 20 |
| LS-C Babble | LS-other + 5 SNR levels | base, tent, suta, ttl | 20 |
| **Total** | | | **52 runs** |

### 1d. Add Common Voice English dataset (~0.5 day)
- Add to `DATASET_CONFIGS` in `src/data.py`: `"common_voice_en"` → `mozilla-foundation/common_voice_17_0`, split `test`, text_key `sentence`
- Run base, tent, suta, ttl on it (4 runs)

---

## Phase 2: Ablations + Model Scaling (Weeks 3-5)

### 2a. Model scaling across Whisper sizes (~1 day GPU)
- Models: whisper-{tiny, base, small, medium, large-v3}
- Dataset: TEDLIUM
- Methods: base, tent, ttl
- 15 runs; use `--fp16` for medium and large-v3
- **Script**: `scripts/exp5_model_scaling.sh`

### 2b. LoRA ablations (~1 day GPU)
- **LoRA rank**: r = {2, 4, 8, 16, 32} on TEDLIUM (5 runs)
- **LoRA placement**: encoder-only, decoder-only, both — needs `--lora_placement` flag
  - Modify `src/models.py` `apply_lora()` to filter targets by prefix
  - 3 runs
- **Learning rate**: lr = {1e-4, 2e-4, 5e-4, 1e-3, 2e-3} on TEDLIUM (5 runs)

### 2c. Entropy distribution analysis (~0.5 day)
- Save per-sample raw entropy values during adaptation (add to `stats` in `ttl.py`)
- Plot histogram: shows why exp(entropy) clusters in [1.0, 2.0]
- This explains why TLM's sample selection doesn't transfer to ASR
- Key figure for the paper's analysis section

### 2d. Statistical significance testing (~0.5 day)
- Bootstrap resampling on WER (standard in ASR papers)
- Use `jiwer` or custom bootstrap over per-utterance errors
- Report p-values for TTL vs Tent, TTL vs SUTA comparisons

---

## Phase 3: Differentiation (Weeks 5-7) — for strong submission

### 3a. Qwen3-ASR experiments (HIGH RISK, ~3-4 days)
- Install `qwen_asr` package (`pip install qwen-asr`)
- **New files**: `src/models_qwen.py`, `src/ttl_qwen.py`
- Apply LoRA to Qwen3 decoder (q_proj, v_proj — standard Qwen3 LLM architecture)
- Test on Qwen3-ASR-0.6B first (fits easily in 40GB)
- Datasets: LS-other, TEDLIUM
- **If this works**: headline result — first TTA on Qwen3-ASR. Nobody has done this.
- **If this fails**: paper stands on Whisper alone (still strong)

### 3b. SGEM baseline (MEDIUM effort, ~2 days)
- Beam search + generalized entropy (Renyi entropy) + negative sampling
- Reference: SGEM GitHub (`drumpt/SGEM`)
- If too complex: cite their numbers, note different eval conditions

### 3c. Additional experiments (if time permits)
- Earnings-22 (financial domain shift)
- Multi-epoch adaptation (1, 2, 3 epochs)
- Online/per-utterance mode (SUTA-style, reset per sample)

---

## Code Changes Summary

### New files
| File | Purpose | Phase |
|------|---------|-------|
| `src/suta.py` | SUTA baseline adapter | 1 |
| `src/noise.py` | Noise injection for LS-C | 1 |
| `src/models_qwen.py` | Qwen3-ASR model loading | 3 |
| `src/ttl_qwen.py` | TTL adapter for Qwen3-ASR | 3 |
| `src/sgem.py` | SGEM baseline | 3 |
| `scripts/exp4_librispeech_c.sh` | LS-C experiment script | 1 |
| `scripts/exp5_model_scaling.sh` | Whisper size sweep | 2 |
| `scripts/exp6_ablations.sh` | LoRA rank/LR/placement | 2 |

### Modified files
| File | Change | Phase |
|------|--------|-------|
| `run_experiment.py` | Add suta/sgem methods, noise args, lora_placement, model_family | 1-3 |
| `src/data.py` | Add Common Voice, Earnings-22 configs; NoisyWhisperCollator | 1 |
| `src/models.py` | Add lora_placement (encoder/decoder/both) filter | 2 |
| `src/ttl.py` | Log per-sample raw entropy values | 2 |
| `analysis/plot_results.py` | Add LS-C plot, scaling plot, entropy histogram | 2 |

### Existing reusable code
- `src/tent.py` — template for SUTA (same entropy loss, add MCC)
- `src/ttl.py` — template for Qwen3-ASR TTL adapter
- `src/data.py` `WhisperCollator` — base for `NoisyWhisperCollator`
- `src/models.py` `apply_lora()` — extend with placement filter
- `analysis/plot_results.py` — extend with new plot types

---

## Key Paper Questions Answered

**Is 9.3% relative significant?** Yes for unsupervised TTA. But paper strength comes from: (a) framework analysis (why CE fails), (b) breadth (noise + domain shift + model scaling), (c) SUTA/SGEM comparison, (d) optionally Qwen3-ASR novelty.

**How to explain the method?** TLM minimizes input perplexity for LLMs. For encoder-decoder ASR, input is audio (no text), so the analogue is decoder output entropy — the model's uncertainty when transcribing. LoRA provides parameter-efficient adaptation without catastrophic forgetting, outperforming LayerNorm-only (Tent) because it has higher capacity in the attention layers where uncertainty concentrates.

**Why doesn't sample selection transfer?** ASR decoder entropy is compressed to [1.0, 2.0] (model is near-deterministic on pseudo-labels), unlike LLM perplexity spanning [7, 400+]. Any P0 threshold either filters everything or nothing. This is a genuine analysis finding.

**Skip Voxtral**: Requires transformers v5.2.0+ (incompatible with our <4.46 pin). Too risky, not worth it.
