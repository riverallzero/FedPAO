from __future__ import annotations

from statistics import mean
from typing import Any

import numpy as np

from src.data.loader import TaskExample
from src.evaluation.metrics import score_prediction
from src.orchestrators.base import OrchestratorEnv


class LocalClient:
    def __init__(self, client_config: dict[str, Any], orchestrator: OrchestratorEnv):
        self.client_config = client_config
        self.orchestrator = orchestrator

    def evaluate(
        self,
        examples: list[TaskExample],
        prompt_latent: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rows = []
        for example in examples:
            result = self.orchestrator.run(example, prompt_latent)
            score = score_prediction(
                example.dataset,
                result.final_answer,
                example.answer,
            )
            row = {
                "client_id": self.client_config["id"],
                "mas": self.client_config["mas"],
                "dataset": self.client_config["dataset"],
                "task_id": example.task_id,
                "split": example.split,
                "final_answer": result.final_answer,
                "prediction": score["prediction"],
                "ground_truth": score["ground_truth"],
                "correct": score["correct"],
                "accuracy": score["accuracy"],
                **result.metrics,
            }
            rows.append(row)

        summary = {
            "client_id": self.client_config["id"],
            "mas": self.client_config["mas"],
            "dataset": self.client_config["dataset"],
            "num_examples": len(rows),
            "accuracy": mean([row["accuracy"] for row in rows]) if rows else 0.0,
            "avg_latency_sec": mean([row["latency_sec"] for row in rows]) if rows else 0.0,
            "avg_total_tokens": mean([row["total_tokens"] for row in rows]) if rows else 0.0,
            "avg_agent_calls": mean([row["agent_calls"] for row in rows]) if rows else 0.0,
            "avg_tool_calls": mean([row["tool_calls"] for row in rows]) if rows else 0.0,
            "redundant_call_rate": mean([row["redundant_call"] for row in rows]) if rows else 0.0,
        }
        return summary, rows

    def loss_from_summary(self, summary: dict[str, Any], objective_config: dict[str, Any]) -> float:
        task_loss = 1.0 - float(summary["accuracy"])
        return (
            float(objective_config.get("task_loss_weight", 1.0)) * task_loss
            + float(objective_config.get("call_cost_weight", 0.0))
            * float(summary["avg_agent_calls"])
            + float(objective_config.get("token_cost_weight", 0.0))
            * float(summary["avg_total_tokens"])
            + float(objective_config.get("latency_weight", 0.0))
            * float(summary["avg_latency_sec"])
            + float(objective_config.get("redundant_call_weight", 0.0))
            * float(summary["redundant_call_rate"])
        )

    def evaluate_loss(
        self,
        examples: list[TaskExample],
        prompt_latent: np.ndarray | None,
        objective_config: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        summary, _ = self.evaluate(examples, prompt_latent)
        loss = self.loss_from_summary(summary, objective_config)
        summary = {**summary, "loss": loss}
        return loss, summary
