from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import cma
import numpy as np


ObjectiveFn = Callable[[np.ndarray], tuple[float, dict[str, Any]]]


@dataclass(frozen=True)
class CmaEsResult:
    best_latent: np.ndarray
    best_loss: float
    best_metrics: dict[str, Any]
    num_evaluations: int
    history: list[dict[str, Any]]


class CmaEsOptimizer:
    def __init__(
        self,
        sigma0: float,
        population_size: int,
        local_iterations: int,
        seed: int = 42,
    ):
        self.sigma0 = sigma0
        self.population_size = population_size
        self.local_iterations = local_iterations
        self.seed = seed

    def optimize(self, initial_latent: np.ndarray, objective: ObjectiveFn) -> CmaEsResult:
        options = {
            "popsize": self.population_size,
            "seed": self.seed,
            "verbose": -9,
        }
        strategy = cma.CMAEvolutionStrategy(
            initial_latent.astype(float).tolist(),
            self.sigma0,
            options,
        )
        best_latent = initial_latent.astype(np.float32, copy=True)
        best_loss, best_metrics = objective(best_latent)
        best_loss = float(best_loss)
        best_metrics = dict(best_metrics)
        history: list[dict[str, Any]] = [
            {
                "iteration": -1,
                "min_loss": best_loss,
                "mean_loss": best_loss,
                "best_loss": best_loss,
                "is_initial": True,
            }
        ]
        num_evaluations = 1

        for iteration in range(self.local_iterations):
            candidates = strategy.ask()
            losses = []
            for candidate in candidates:
                latent = np.asarray(candidate, dtype=np.float32)
                loss, metrics = objective(latent)
                losses.append(loss)
                num_evaluations += 1
                if loss < best_loss:
                    best_latent = latent.copy()
                    best_loss = float(loss)
                    best_metrics = dict(metrics)

            strategy.tell(candidates, losses)
            history.append(
                {
                    "iteration": iteration,
                    "min_loss": float(min(losses)),
                    "mean_loss": float(np.mean(losses)),
                    "best_loss": best_loss,
                }
            )

        return CmaEsResult(
            best_latent=best_latent,
            best_loss=best_loss,
            best_metrics=best_metrics,
            num_evaluations=num_evaluations,
            history=history,
        )
