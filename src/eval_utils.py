"""Evaluation helpers: WER computation and full-set evaluation."""

import re
import torch
from jiwer import wer as compute_wer
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@torch.no_grad()
def evaluate(model, processor, dataloader, device: str = "cuda", language: str = "en"):
    """Run greedy decoding on *dataloader* and return WER.

    Returns (wer, predictions, references).
    """
    model.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    all_preds = []
    all_refs = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_features = batch["input_features"].to(device)
        refs = batch["references"]

        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

        all_preds.extend(decoded)
        all_refs.extend(refs)

    # Normalize both sides
    norm_preds = [normalize_text(p) for p in all_preds]
    norm_refs = [normalize_text(r) for r in all_refs]

    # Filter out empty references (some datasets have them)
    pairs = [(p, r) for p, r in zip(norm_preds, norm_refs) if r]
    if not pairs:
        return 1.0, all_preds, all_refs

    norm_preds, norm_refs = zip(*pairs)
    word_error_rate = compute_wer(list(norm_refs), list(norm_preds))
    return word_error_rate, all_preds, all_refs
