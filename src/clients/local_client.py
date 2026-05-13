from __future__ import annotations

from statistics import mean
from typing import Any

import numpy as np

from src.data.loader import TaskExample
from src.evaluation.normalization import normalize_answer
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
            "invalid_action_rate": mean([row["invalid_action"] for row in rows]) if rows else 0.0,
            "invalid_tool_call_count": sum([row["invalid_tool_call_count"] for row in rows]),
            "premature_termination_rate": mean([row["premature_termination"] for row in rows]) if rows else 0.0,
            "late_termination_rate": mean([row["late_termination"] for row in rows]) if rows else 0.0,
        }
        return summary, rows

    def loss_from_summary(self, summary: dict[str, Any], objective_config: dict[str, Any]) -> float:
        task_loss = float(summary.get("task_loss", 1.0 - float(summary["accuracy"])))
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
            + float(objective_config.get("invalid_action_weight", 0.0))
            * float(summary["invalid_action_rate"])
        )

    def evaluate_dense_loss(
        self,
        examples: list[TaskExample],
        prompt_latent: np.ndarray | None,
        objective_config: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        score_mode = objective_config.get("score_mode", "exact_match")
        rows = []

        for example in examples:
            if (
                example.dataset == "mmlu_pro"
                and score_mode in {"mmlu_choice_logprob", "hybrid_logprob"}
                and hasattr(self.orchestrator, "score_mmlu_choices")
            ):
                result = self.orchestrator.score_mmlu_choices(example, prompt_latent)
                ground_truth = normalize_answer(example.dataset, example.answer)
                prediction = result["prediction"]
                correct_logprob = result["choice_logprobs"].get(ground_truth)
                if correct_logprob is None:
                    correct_logprob = min(result["choice_logprobs"].values())
                correct = prediction == ground_truth
                row = {
                    "client_id": self.client_config["id"],
                    "mas": self.client_config["mas"],
                    "dataset": self.client_config["dataset"],
                    "task_id": example.task_id,
                    "split": example.split,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "correct": correct,
                    "accuracy": 1.0 if correct else 0.0,
                    "task_loss": -float(correct_logprob),
                    **result["metrics"],
                }
            elif (
                score_mode == "hybrid_logprob"
                and hasattr(self.orchestrator, "score_answer")
            ):
                result = self.orchestrator.score_answer(example, prompt_latent)
                row = {
                    "client_id": self.client_config["id"],
                    "mas": self.client_config["mas"],
                    "dataset": self.client_config["dataset"],
                    "task_id": example.task_id,
                    "split": example.split,
                    "prediction": "",
                    "ground_truth": normalize_answer(example.dataset, example.answer),
                    "correct": False,
                    "accuracy": 0.0,
                    "task_loss": -float(result["answer_logprob"]),
                    **result["metrics"],
                }
            else:
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
                    "prediction": score["prediction"],
                    "ground_truth": score["ground_truth"],
                    "correct": score["correct"],
                    "accuracy": score["accuracy"],
                    "task_loss": 1.0 - score["accuracy"],
                    **result.metrics,
                }
            rows.append(row)

        summary = self.summarize_rows(rows)
        return summary, rows

    def summarize_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "client_id": self.client_config["id"],
            "mas": self.client_config["mas"],
            "dataset": self.client_config["dataset"],
            "num_examples": len(rows),
            "accuracy": mean([row["accuracy"] for row in rows]) if rows else 0.0,
            "task_loss": mean([row["task_loss"] for row in rows]) if rows and "task_loss" in rows[0] else 0.0,
            "avg_latency_sec": mean([row["latency_sec"] for row in rows]) if rows else 0.0,
            "avg_total_tokens": mean([row["total_tokens"] for row in rows]) if rows else 0.0,
            "avg_agent_calls": mean([row["agent_calls"] for row in rows]) if rows else 0.0,
            "avg_tool_calls": mean([row["tool_calls"] for row in rows]) if rows else 0.0,
            "redundant_call_rate": mean([row["redundant_call"] for row in rows]) if rows else 0.0,
            "invalid_action_rate": mean([row["invalid_action"] for row in rows]) if rows else 0.0,
            "invalid_tool_call_count": sum([row["invalid_tool_call_count"] for row in rows]),
            "premature_termination_rate": mean([row["premature_termination"] for row in rows]) if rows else 0.0,
            "late_termination_rate": mean([row["late_termination"] for row in rows]) if rows else 0.0,
        }

    def evaluate_loss(
        self,
        examples: list[TaskExample],
        prompt_latent: np.ndarray | None,
        objective_config: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        if objective_config.get("score_mode", "exact_match") == "exact_match":
            summary, _ = self.evaluate(examples, prompt_latent)
        else:
            summary, _ = self.evaluate_dense_loss(
                examples,
                prompt_latent,
                objective_config,
            )
        loss = self.loss_from_summary(summary, objective_config)
        summary = {**summary, "loss": loss}
        return loss, summary
