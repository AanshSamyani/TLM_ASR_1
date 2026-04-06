"""On-the-fly noise injection for LibriSpeech-C style corruption.

Supported noise types:
  - gaussian: additive white Gaussian noise
  - babble:   sum of N random speech utterances from the dataset

All noise is calibrated to a target SNR (signal-to-noise ratio in dB).
"""

import numpy as np


def _signal_power(audio: np.ndarray) -> float:
    """Mean squared amplitude."""
    return float(np.mean(audio.astype(np.float64) ** 2))


def add_gaussian_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at the specified SNR (dB)."""
    sig_power = _signal_power(audio)
    if sig_power < 1e-10:
        return audio
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
    return (audio + noise).astype(audio.dtype)


def add_babble_noise(
    audio: np.ndarray,
    snr_db: float,
    dataset,
    n_babble: int = 3,
) -> np.ndarray:
    """Add babble noise by mixing N random utterances from *dataset*.

    Each babble source is padded/truncated to match *audio* length,
    then the mixture is scaled to the target SNR.
    """
    if dataset is None or len(dataset) < 2:
        return add_gaussian_noise(audio, snr_db)

    indices = np.random.choice(len(dataset), size=n_babble, replace=True)
    babble = np.zeros(len(audio), dtype=np.float64)
    for idx in indices:
        other = dataset[int(idx)]["audio"]["array"].astype(np.float64)
        if len(other) >= len(audio):
            babble += other[: len(audio)]
        else:
            babble[: len(other)] += other
    babble /= n_babble

    # Scale babble to target SNR
    sig_power = _signal_power(audio)
    noise_power = _signal_power(babble)
    if noise_power < 1e-10 or sig_power < 1e-10:
        return audio
    scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
    noisy = audio.astype(np.float64) + babble * scale
    return noisy.astype(audio.dtype)
