"""Test-Time Learning (TTL) adaptation for Whisper ASR.

Adapted from the TLM paper (arXiv 2505.20633) for encoder-decoder ASR:

For LLMs, TLM minimises *input perplexity* P(x; Theta) — the model's
uncertainty about the input text.  For Whisper (encoder-decoder), the input
is audio, so the direct analogue is the decoder's output uncertainty.

Three perplexity / loss modes are available:

  ppl_method="ce"       — CE loss against own pseudo-labels (weak baseline)
  ppl_method="entropy"  — mean token entropy of decoder output (recommended)
  ppl_method="gen"      — generation-time perplexity for sample selection,
                          entropy loss for gradient updates

Steps:
  1. Generate pseudo-transcriptions via greedy decoding
  2. Compute loss (CE or entropy) via teacher-forced forward pass
  3. Optionally weight the loss by a perplexity-based sample selector
  4. Back-propagate through LoRA parameters only
"""

import math
import torch
from tqdm import tqdm


class TTLAdapter:
    """Adapt a LoRA-wrapped Whisper model on unlabelled test audio."""

    def __init__(
        self,
        model,
        processor,
        lr: float = 5e-5,
        sample_selector=None,
        device: str = "cuda",
        language: str = "en",
        ppl_method: str = "entropy",
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.sample_selector = sample_selector
        self.ppl_method = ppl_method
        self.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )
        self.n_forced = len(self.forced_decoder_ids) if self.forced_decoder_ids else 0

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=lr)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_pseudo_labels(self, input_features: torch.Tensor):
        """Generate pseudo-labels via greedy decoding.

        Returns (sequences, gen_ppl) where gen_ppl is the generation-time
        perplexity (only computed when ppl_method="gen", else None).
        """
        self.model.eval()

        if self.ppl_method == "gen":
            outputs = self.model.generate(
                input_features=input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                max_new_tokens=225,
                output_scores=True,
                return_dict_in_generate=True,
            )
            sequences = outputs.sequences
            scores = outputs.scores  # tuple of (batch, vocab) per gen step

            # Per-token log-probs for content tokens (skip forced prefix)
            log_probs = []
            for t in range(self.n_forced, len(scores)):
                log_p = torch.log_softmax(scores[t], dim=-1)
                token_idx = sequences[:, t + 1]  # +1 for decoder_start_token
                tok_lp = log_p.gather(-1, token_idx.unsqueeze(-1)).squeeze(-1)
                log_probs.append(tok_lp)

            if log_probs:
                log_probs_t = torch.stack(log_probs, dim=1)
                ppl_per_sample = torch.exp(-log_probs_t.mean(dim=1))
                gen_ppl = ppl_per_sample.mean().item()
            else:
                gen_ppl = 1.0

            return sequences, gen_ppl
        else:
            sequences = self.model.generate(
                input_features=input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                max_new_tokens=225,
            )
            return sequences, None

    def _ce_loss(self, input_features: torch.Tensor, pseudo_labels: torch.Tensor):
        """Teacher-forced CE loss against pseudo-labels.  Returns (loss, ppl)."""
        self.model.eval()
        labels = pseudo_labels[:, 1:].clone()
        # Call through base_model to bypass PeftModelForSeq2SeqLM.forward()
        # which injects input_ids that Whisper doesn't accept.
        outputs = self.model.base_model(input_features=input_features, labels=labels)
        loss = outputs.loss
        ppl = math.exp(min(loss.item(), 100))
        return loss, ppl

    def _entropy_loss(self, input_features: torch.Tensor, pseudo_labels: torch.Tensor):
        """Mean token entropy of the decoder output distribution.

        This is the correct ASR analogue of the TLM paper's input perplexity
        minimisation: it measures and reduces the model's uncertainty.

        Returns (entropy_loss, exp_entropy_as_ppl).
        """
        self.model.eval()
        decoder_input_ids = pseudo_labels[:, :-1]
        # Call through base_model to bypass PeftModelForSeq2SeqLM.forward()
        outputs = self.model.base_model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)
        mean_entropy = entropy.mean()
        ppl = math.exp(min(mean_entropy.item(), 100))
        return mean_entropy, ppl

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def adapt(self, dataloader, n_epochs: int = 1):
        """One-pass (offline) adaptation over the test data.

        Returns a dict of statistics for logging / plotting.
        """
        stats = {
            "total_samples": 0,
            "adapted_samples": 0,
            "skipped_samples": 0,
            "perplexities": [],
            "losses": [],
            "raw_entropies": [],
        }

        for epoch in range(n_epochs):
            pbar = tqdm(dataloader, desc=f"TTL adapt epoch {epoch + 1}/{n_epochs}")
            for batch in pbar:
                input_features = batch["input_features"].to(self.device)
                stats["total_samples"] += input_features.shape[0]

                # Step 1 — pseudo-labels (+ gen perplexity if ppl_method="gen")
                pseudo_labels, gen_ppl = self._generate_pseudo_labels(input_features)

                # Skip if pseudo-label is trivially short (only forced tokens)
                if pseudo_labels.shape[1] <= 5:
                    stats["skipped_samples"] += 1
                    continue

                # Step 2 — compute loss & perplexity
                if self.ppl_method == "ce":
                    loss, ppl = self._ce_loss(input_features, pseudo_labels)
                elif self.ppl_method == "entropy":
                    loss, ppl = self._entropy_loss(input_features, pseudo_labels)
                elif self.ppl_method == "gen":
                    # Generation perplexity for selection, entropy for gradient
                    loss, _ = self._entropy_loss(input_features, pseudo_labels)
                    ppl = gen_ppl
                else:
                    raise ValueError(f"Unknown ppl_method: {self.ppl_method}")

                stats["perplexities"].append(ppl)
                if self.ppl_method in ("entropy", "gen"):
                    stats["raw_entropies"].append(loss.item())

                # Step 3 — sample selection
                if self.sample_selector is not None:
                    weight = self.sample_selector.compute_weight(ppl)
                    if weight == 0.0:
                        stats["skipped_samples"] += 1
                        pbar.set_postfix(ppl=f"{ppl:.1f}", status="skip")
                        continue
                else:
                    weight = 1.0

                # Step 4 — gradient update through LoRA
                weighted_loss = weight * loss
                self.optimizer.zero_grad()
                weighted_loss.backward()
                self.optimizer.step()

                stats["adapted_samples"] += 1
                stats["losses"].append(loss.item())
                pbar.set_postfix(ppl=f"{ppl:.1f}", loss=f"{loss.item():.3f}")

        return stats
