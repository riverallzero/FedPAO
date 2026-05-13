from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SoftPromptProjector:
    latent_dim: int
    prompt_length: int
    hidden_size: int
    init_std: float = 0.02
    seed: int = 42

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        projection = rng.normal(
            loc=0.0,
            scale=self.init_std,
            size=(self.latent_dim, self.prompt_length * self.hidden_size),
        )
        object.__setattr__(
            self,
            "projection",
            torch.tensor(projection, dtype=torch.float32),
        )

    def project(
        self,
        latent: np.ndarray | torch.Tensor,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if isinstance(latent, np.ndarray):
            z = torch.from_numpy(latent.astype(np.float32, copy=False))
        else:
            z = latent.detach().to(dtype=torch.float32)
        if z.ndim != 1 or z.shape[0] != self.latent_dim:
            raise ValueError(f"Expected latent shape ({self.latent_dim},), got {tuple(z.shape)}")

        prompt = z @ self.projection
        prompt = prompt.reshape(self.prompt_length, self.hidden_size)
        if device is not None or dtype is not None:
            prompt = prompt.to(device=device, dtype=dtype)
        return prompt

    def projection_hash(self) -> str:
        return hashlib.sha1(self.projection.numpy().tobytes()).hexdigest()[:16]
