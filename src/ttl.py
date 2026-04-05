"""Test-Time Learning (TTL) adaptation for Whisper ASR.

Core idea (adapted from the TLM paper for LLMs):
  1. Generate pseudo-transcriptions via greedy decoding  (no labels needed)
  2. Compute CE loss of the decoder teacher-forced on those pseudo-labels
     (this is the speech analogue of "input perplexity minimisation")
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
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.sample_selector = sample_selector
        self.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=lr)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_pseudo_labels(self, input_features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model.generate(
            input_features=input_features,
            forced_decoder_ids=self.forced_decoder_ids,
            max_new_tokens=225,
        )

    def _ce_loss(self, input_features: torch.Tensor, pseudo_labels: torch.Tensor):
        """Teacher-forced CE loss.  Returns (loss, perplexity)."""
        self.model.train()
        # labels = pseudo_labels without the leading decoder_start_token_id;
        # the model's forward() prepends it internally via shift_tokens_right.
        labels = pseudo_labels[:, 1:].clone()
        # Call through base_model to bypass PeftModelForSeq2SeqLM.forward()
        # which injects input_ids that Whisper doesn't accept.
        # LoRA layers remain active as they're injected into the model weights.
        outputs = self.model.base_model(input_features=input_features, labels=labels)
        loss = outputs.loss
        ppl = math.exp(min(loss.item(), 100))  # clamp to avoid overflow
        return loss, ppl

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
        }

        for epoch in range(n_epochs):
            pbar = tqdm(dataloader, desc=f"TTL adapt epoch {epoch + 1}/{n_epochs}")
            for batch in pbar:
                input_features = batch["input_features"].to(self.device)
                stats["total_samples"] += input_features.shape[0]

                # Step 1 — pseudo-labels
                pseudo_labels = self._generate_pseudo_labels(input_features)

                # Skip if pseudo-label is trivially short (only forced tokens)
                if pseudo_labels.shape[1] <= 5:
                    stats["skipped_samples"] += 1
                    continue

                # Step 2 — CE loss & perplexity
                loss, ppl = self._ce_loss(input_features, pseudo_labels)
                stats["perplexities"].append(ppl)

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
