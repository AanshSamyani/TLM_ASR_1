"""Tent baseline: test-time entropy minimisation for Whisper.

Reference — Wang et al., "Tent: Fully Test-Time Adaptation by Entropy
Minimization", ICLR 2021.

For an encoder-decoder model the approach is:
  1. Generate pseudo-labels via greedy decoding (needed for teacher forcing).
  2. Run a teacher-forced forward pass.
  3. Compute the mean entropy of the decoder's output distribution.
  4. Back-propagate through LayerNorm parameters only.
"""

import math
import torch
from tqdm import tqdm


class TentAdapter:
    """Adapt a Whisper model at test time via entropy minimisation."""

    def __init__(
        self,
        model,
        processor,
        lr: float = 1e-3,
        ln_params: list = None,
        device: str = "cuda",
        language: str = "en",
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

        if ln_params is None or len(ln_params) == 0:
            raise ValueError("No LayerNorm parameters provided for Tent.")
        self.optimizer = torch.optim.Adam(ln_params, lr=lr)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_pseudo_labels(self, input_features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model.generate(
            input_features=input_features,
            forced_decoder_ids=self.forced_decoder_ids,
            max_new_tokens=225,
        )

    @staticmethod
    def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
        """Mean token-level entropy of the decoder output distribution."""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)
        return entropy.mean()

    # ------------------------------------------------------------------
    def adapt(self, dataloader, n_epochs: int = 1):
        """Entropy-minimisation pass over the test data."""
        stats = {
            "total_samples": 0,
            "adapted_samples": 0,
            "skipped_samples": 0,
            "perplexities": [],
            "losses": [],
        }

        for epoch in range(n_epochs):
            pbar = tqdm(dataloader, desc=f"Tent adapt epoch {epoch + 1}/{n_epochs}")
            for batch in pbar:
                input_features = batch["input_features"].to(self.device)
                stats["total_samples"] += input_features.shape[0]

                # Pseudo-labels for teacher-forced forward pass
                pseudo_labels = self._generate_pseudo_labels(input_features)
                if pseudo_labels.shape[1] <= 5:
                    stats["skipped_samples"] += 1
                    continue

                # Teacher-forced forward — need decoder_input_ids
                decoder_input_ids = pseudo_labels[:, :-1]  # drop eos

                self.model.train()
                outputs = self.model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                )
                loss = self._entropy_loss(outputs.logits)

                # Also track CE-equivalent perplexity for comparability
                with torch.no_grad():
                    labels = pseudo_labels[:, 1:]
                    ce_out = self.model(input_features=input_features, labels=labels)
                    ppl = math.exp(min(ce_out.loss.item(), 100))
                stats["perplexities"].append(ppl)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                stats["adapted_samples"] += 1
                stats["losses"].append(loss.item())
                pbar.set_postfix(ent=f"{loss.item():.3f}", ppl=f"{ppl:.1f}")

        return stats
