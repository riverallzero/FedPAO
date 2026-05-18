from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from src.data.loader import TaskExample


@dataclass(frozen=True)
class OrchestratorResult:
    trajectory: list[dict[str, Any]]
    final_answer: str
    metrics: dict[str, float | int]
    metadata: dict[str, Any] = field(default_factory=dict)


class OrchestratorEnv(Protocol):
    def run(
        self,
        example: TaskExample,
        prompt_latent: np.ndarray | None = None,
    ) -> OrchestratorResult:
        ...
