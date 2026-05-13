from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    latency_sec: float


class LLMBackend(Protocol):
    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        ...

    def generate_with_soft_prompt(
        self,
        prompt: str,
        soft_prompt_embeds: torch.Tensor,
        max_new_tokens: int,
    ) -> GenerationResult:
        ...

