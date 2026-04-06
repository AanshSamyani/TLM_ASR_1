"""Dataset loading and batching utilities for ASR experiments."""

import numpy as np
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader

from src.noise import add_gaussian_noise, add_babble_noise


# Each entry maps a short name to HuggingFace load_dataset arguments.
DATASET_CONFIGS = {
    "librispeech_clean": {
        "path": "librispeech_asr",
        "name": "clean",
        "split": "test",
        "text_key": "text",
    },
    "librispeech_other": {
        "path": "librispeech_asr",
        "name": "other",
        "split": "test",
        "text_key": "text",
    },
    "tedlium": {
        "path": "distil-whisper/tedlium",
        "name": "release3",
        "split": "test",
        "text_key": "text",
    },
    "common_voice_en": {
        "path": "mozilla-foundation/common_voice_17_0",
        "name": "en",
        "split": "test",
        "text_key": "sentence",
    },
}


def load_asr_dataset(dataset_name: str, max_samples: int = None):
    """Load an ASR dataset and resample audio to 16 kHz.

    Returns (HF Dataset, text_column_name).
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[dataset_name]
    ds = load_dataset(
        cfg["path"],
        cfg["name"],
        split=cfg["split"],
        trust_remote_code=True,
    )
    # Guarantee 16 kHz mono audio
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    print(f"Loaded {dataset_name}: {len(ds)} samples")
    return ds, cfg["text_key"]


class WhisperCollator:
    """Collate a list of dataset rows into a batch for Whisper."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        audio_arrays = [item["audio"]["array"] for item in batch]
        texts = [item.get("text", item.get("sentence", "")) for item in batch]

        features = self.processor(
            audio_arrays,
            sampling_rate=16_000,
            return_tensors="pt",
        )
        return {
            "input_features": features.input_features,
            "references": texts,
        }


class NoisyWhisperCollator(WhisperCollator):
    """Wraps WhisperCollator, injecting noise into audio before feature extraction."""

    def __init__(self, processor, noise_type: str = "gaussian", snr_db: float = 10.0,
                 dataset=None, n_babble: int = 3):
        super().__init__(processor)
        self.noise_type = noise_type
        self.snr_db = snr_db
        self.dataset = dataset  # needed for babble noise sources
        self.n_babble = n_babble

    def __call__(self, batch):
        for item in batch:
            audio = item["audio"]["array"]
            if self.noise_type == "gaussian":
                item["audio"]["array"] = add_gaussian_noise(audio, self.snr_db)
            elif self.noise_type == "babble":
                item["audio"]["array"] = add_babble_noise(
                    audio, self.snr_db, self.dataset, self.n_babble,
                )
        return super().__call__(batch)


def create_dataloader(dataset, processor, batch_size: int = 1, shuffle: bool = False,
                      noise_type: str = "none", snr_db: float = 10.0):
    """Build a DataLoader that yields Whisper-ready batches.

    If *noise_type* is not "none", audio is corrupted on-the-fly.
    """
    if noise_type != "none":
        collator = NoisyWhisperCollator(
            processor, noise_type=noise_type, snr_db=snr_db, dataset=dataset,
        )
    else:
        collator = WhisperCollator(processor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
