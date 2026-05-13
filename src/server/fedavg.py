from __future__ import annotations

import numpy as np


def fedavg(latents: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    if not latents:
        raise ValueError("fedavg requires at least one latent")

    stacked = np.stack([latent.astype(np.float32, copy=False) for latent in latents])
    if weights is None:
        return stacked.mean(axis=0).astype(np.float32)

    weight_array = np.asarray(weights, dtype=np.float32)
    if weight_array.shape != (len(latents),):
        raise ValueError("weights must have one value per latent")
    weight_sum = float(weight_array.sum())
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value")
    normalized = weight_array / weight_sum
    return np.sum(stacked * normalized[:, None], axis=0).astype(np.float32)


def communication_bytes(latent: np.ndarray) -> int:
    return int(latent.astype(np.float32, copy=False).nbytes)

