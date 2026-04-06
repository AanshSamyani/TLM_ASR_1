"""SUTA baseline: test-time adaptation for ASR via entropy + MCC.

Reference — Lin et al., "Listen, Adapt, Better WER: Source-free
Single-utterance Test-time Adaptation for ASR", INTERSPEECH 2022.

Key differences from Tent:
  - Per-utterance reset: model params are restored after each utterance
  - Multi-step optimisation: K gradient steps per utterance (default 10)
  - MCC loss: minimum class confusion regulariser (top-k approximation
    for large-vocabulary models like Whisper)
  - Temperature smoothing on logits (T=2.0)

Combined loss:  L = alpha * H(y) + (1 - alpha) * L_mcc(y)
"""

import math
import re

import torch
from jiwer import wer as compute_wer
from tqdm import tqdm


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SutaAdapter:
    """Adapt a Whisper model per-utterance with entropy + MCC."""

    def __init__(
        self,
        model,
        processor,
        lr: float = 1e-3,
        ln_params: list = None,
        device: str = "cuda",
        language: str = "en",
        suta_steps: int = 10,
        suta_alpha: float = 0.5,
        temperature: float = 2.0,
        mcc_topk: int = 100,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.suta_steps = suta_steps
        self.suta_alpha = suta_alpha
        self.temperature = temperature
        self.mcc_topk = mcc_topk
        self.lr = lr
        self.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

        if ln_params is None or len(ln_params) == 0:
            raise ValueError("No LayerNorm parameters provided for SUTA.")
        self.ln_params = ln_params

        # Save initial state for per-utterance reset
        self._initial_state = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._initial_state[name] = param.data.clone()

    # ------------------------------------------------------------------
    def _reset_model(self):
        """Restore trainable parameters to their initial values."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._initial_state:
                    param.data.copy_(self._initial_state[name])

    @torch.no_grad()
    def _generate_pseudo_labels(self, input_features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model.generate(
            input_features=input_features,
            forced_decoder_ids=self.forced_decoder_ids,
            max_new_tokens=225,
        )

    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Mean token entropy with temperature smoothing."""
        logits_t = logits / self.temperature
        probs = torch.softmax(logits_t, dim=-1)
        log_probs = torch.log_softmax(logits_t, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)
        return entropy.mean()

    def _mcc_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Minimum class confusion loss (top-k approximation).

        Computes a class correlation matrix from the softmax distribution
        and minimises its row-wise entropy, encouraging the model to
        produce peaked, non-confused predictions.
        """
        logits_t = logits / self.temperature
        # Top-k to keep memory manageable for large vocab (51K tokens)
        topk_logits, _ = logits_t.topk(self.mcc_topk, dim=-1)
        probs = torch.softmax(topk_logits, dim=-1)  # (1, seq_len, k)
        P = probs.squeeze(0)  # (seq_len, k)

        # Correlation matrix: C = P^T @ P / seq_len
        C = P.T @ P / P.shape[0]  # (k, k)

        # Row-normalise to get a conditional distribution
        C_hat = C / C.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Row-wise entropy, averaged over classes
        mcc = -(C_hat * torch.log(C_hat.clamp(min=1e-8))).sum(dim=-1).mean()
        return mcc

    # ------------------------------------------------------------------
    def adapt_and_evaluate(self, dataloader):
        """Per-utterance adaptation + decoding.

        For each utterance:
          1. Generate pseudo-labels (greedy decode)
          2. K gradient steps on entropy + MCC loss
          3. Decode with adapted model
          4. Reset model to initial state

        Returns (wer, predictions, references, stats).
        """
        all_preds = []
        all_refs = []
        stats = {
            "total_samples": 0,
            "adapted_samples": 0,
            "skipped_samples": 0,
            "perplexities": [],
            "losses": [],
        }

        pbar = tqdm(dataloader, desc="SUTA adapt+eval")
        for batch in pbar:
            input_features = batch["input_features"].to(self.device)
            refs = batch["references"]
            stats["total_samples"] += input_features.shape[0]

            # Step 1 — pseudo-labels
            pseudo_labels = self._generate_pseudo_labels(input_features)
            if pseudo_labels.shape[1] <= 5:
                # Trivially short — decode without adaptation
                self.model.eval()
                gen_ids = self.model.generate(
                    input_features=input_features,
                    forced_decoder_ids=self.forced_decoder_ids,
                    max_new_tokens=225,
                )
                decoded = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
                all_preds.extend(decoded)
                all_refs.extend(refs)
                stats["skipped_samples"] += 1
                continue

            decoder_input_ids = pseudo_labels[:, :-1]

            # Step 2 — K optimisation steps
            optimizer = torch.optim.Adam(self.ln_params, lr=self.lr)
            last_loss = 0.0
            for _step in range(self.suta_steps):
                self.model.eval()
                outputs = self.model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                )
                e_loss = self._entropy_loss(outputs.logits)
                if self.suta_alpha < 1.0:
                    m_loss = self._mcc_loss(outputs.logits)
                    loss = self.suta_alpha * e_loss + (1 - self.suta_alpha) * m_loss
                else:
                    loss = e_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = loss.item()

            stats["losses"].append(last_loss)
            stats["perplexities"].append(math.exp(min(last_loss, 100)))

            # Step 3 — decode with adapted model
            self.model.eval()
            with torch.no_grad():
                gen_ids = self.model.generate(
                    input_features=input_features,
                    forced_decoder_ids=self.forced_decoder_ids,
                    max_new_tokens=225,
                )
            decoded = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
            all_preds.extend(decoded)
            all_refs.extend(refs)
            stats["adapted_samples"] += 1
            pbar.set_postfix(loss=f"{last_loss:.3f}")

            # Step 4 — reset for next utterance
            self._reset_model()

        # Compute WER
        norm_preds = [_normalize_text(p) for p in all_preds]
        norm_refs = [_normalize_text(r) for r in all_refs]
        pairs = [(p, r) for p, r in zip(norm_preds, norm_refs) if r]
        if not pairs:
            return 1.0, all_preds, all_refs, stats
        norm_preds, norm_refs = zip(*pairs)
        word_error_rate = compute_wer(list(norm_refs), list(norm_preds))

        return word_error_rate, all_preds, all_refs, stats
