"""Model loading and LoRA/Tent parameter setup for Whisper."""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import get_peft_model, LoraConfig, TaskType


def load_whisper(model_name: str = "openai/whisper-small", device: str = "cuda"):
    """Load a pretrained Whisper model and processor."""
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    return model, processor


def apply_lora(
    model,
    rank: int = 8,
    alpha: int = 16,
    target_modules: list = None,
    dropout: float = 0.0,
    placement: str = "both",
):
    """Wrap a Whisper model with LoRA adapters.

    By default targets the query and value projections in all attention
    layers (encoder + decoder).  Returns the PeftModel wrapper.

    Args:
        placement: "both" (default), "encoder", or "decoder".
            When not "both", LoRA is applied everywhere but the
            non-target side's LoRA params are frozen.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, config)

    # Freeze LoRA params on the non-target side
    if placement == "encoder":
        for name, param in model.named_parameters():
            if param.requires_grad and "decoder" in name:
                param.requires_grad = False
    elif placement == "decoder":
        for name, param in model.named_parameters():
            if param.requires_grad and "encoder" in name:
                param.requires_grad = False

    model.print_trainable_parameters()
    return model


def get_layernorm_params(model):
    """Freeze everything except LayerNorm parameters (for Tent).

    Returns the list of LayerNorm parameters with requires_grad=True.
    """
    params = []
    for name, param in model.named_parameters():
        if "layer_norm" in name.lower():
            param.requires_grad = True
            params.append(param)
        else:
            param.requires_grad = False
    n = sum(p.numel() for p in params)
    print(f"Tent: {n:,} trainable LayerNorm parameters")
    return params
