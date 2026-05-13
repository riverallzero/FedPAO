from __future__ import annotations

import numpy as np

from src.data.loader import TaskExample
from src.evaluation.normalization import numeric_answer_variants
from src.models.hf_causal import HFCausalBackend
from src.orchestrators.base import OrchestratorResult
from src.prompts.soft_prompt import SoftPromptProjector


def format_task_prompt(client_config: dict, example: TaskExample) -> str:
    mas = client_config["mas"]

    if example.dataset == "mmlu_pro":
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options = "\n".join(
            f"{letters[idx]}. {option}" for idx, option in enumerate(example.options or [])
        )
        return (
            f"You are the local frozen LLM backend for the {mas} client.\n"
            "Answer the multiple-choice question. Return only the option letter.\n\n"
            f"Question: {example.question}\n"
            f"Options:\n{options}\n"
            "Answer:"
        )

    if example.dataset == "gsm_hard":
        return (
            f"You are the local frozen LLM backend for the {mas} client.\n"
            "Solve the math problem. Return only the final numeric answer.\n\n"
            f"Problem: {example.question}\n"
            "Answer:"
        )

    raise ValueError(f"Unsupported dataset: {example.dataset}")


class MockOrchestrator:
    def __init__(
        self,
        client_config: dict,
        backend: HFCausalBackend,
        projector: SoftPromptProjector | None = None,
        max_new_tokens: int = 32,
    ):
        self.client_config = client_config
        self.backend = backend
        self.projector = projector
        self.max_new_tokens = max_new_tokens

    def run(
        self,
        example: TaskExample,
        prompt_latent: np.ndarray | None = None,
    ) -> OrchestratorResult:
        prompt = format_task_prompt(self.client_config, example)
        if prompt_latent is not None and self.projector is not None:
            soft_prompt = self.projector.project(prompt_latent, device=self.backend.device)
            response = self.backend.generate_with_soft_prompt(
                prompt,
                soft_prompt,
                self.max_new_tokens,
            )
        else:
            response = self.backend.generate(prompt, self.max_new_tokens)

        trajectory = [
            {
                "step_id": 1,
                "selected_component": "llm",
                "capability": "direct_answer",
                "tokens": response.total_tokens,
                "latency_sec": response.latency_sec,
                "is_redundant": False,
            }
        ]
        metrics = {
            "prompt_tokens": response.prompt_tokens,
            "generated_tokens": response.generated_tokens,
            "total_tokens": response.total_tokens,
            "latency_sec": response.latency_sec,
            "agent_calls": 1,
            "tool_calls": 0,
            "redundant_call": 0,
            "invalid_action": 0,
            "invalid_tool_call_count": 0,
            "premature_termination": 0,
            "late_termination": 0,
        }
        return OrchestratorResult(
            trajectory=trajectory,
            final_answer=response.text,
            metrics=metrics,
        )

    def score_mmlu_choices(
        self,
        example: TaskExample,
        prompt_latent: np.ndarray | None = None,
    ) -> dict:
        if example.dataset != "mmlu_pro":
            raise ValueError("score_mmlu_choices only supports mmlu_pro")

        prompt = format_task_prompt(self.client_config, example)
        choices = list("ABCDEFGHIJ")[: len(example.options or [])]
        soft_prompt = None
        if prompt_latent is not None and self.projector is not None:
            soft_prompt = self.projector.project(prompt_latent, device=self.backend.device)

        response = self.backend.score_choices_with_soft_prompt(
            prompt,
            choices,
            soft_prompt,
        )
        prediction = max(response.scores, key=response.scores.get)
        return {
            "prediction": prediction,
            "choice_logprobs": response.scores,
            "metrics": {
                "prompt_tokens": response.prompt_tokens,
                "generated_tokens": 0,
                "total_tokens": response.total_tokens,
                "latency_sec": response.latency_sec,
                "agent_calls": 1,
                "tool_calls": 0,
                "redundant_call": 0,
                "invalid_action": 0,
                "invalid_tool_call_count": 0,
                "premature_termination": 0,
                "late_termination": 0,
            },
        }

    def score_answer(
        self,
        example: TaskExample,
        prompt_latent: np.ndarray | None = None,
    ) -> dict:
        prompt = format_task_prompt(self.client_config, example)
        soft_prompt = None
        if prompt_latent is not None and self.projector is not None:
            soft_prompt = self.projector.project(prompt_latent, device=self.backend.device)

        if example.dataset == "gsm_hard":
            continuations = {}
            for idx, variant in enumerate(numeric_answer_variants(example.answer)):
                continuations[f"variant_{idx}_bare"] = variant
                continuations[f"variant_{idx}_spaced"] = f" {variant}"
            response = self.backend.score_continuations(
                prompt,
                continuations,
                soft_prompt,
            )
            best_key = max(response.scores, key=response.scores.get)
            answer_logprob = response.scores[best_key]
            best_answer = continuations[best_key].strip()
        else:
            response = self.backend.score_answer_with_soft_prompt(
                prompt,
                example.answer,
                soft_prompt,
            )
            answer_logprob = response.scores["answer"]
            best_answer = example.answer
        return {
            "answer_logprob": answer_logprob,
            "scored_answer": best_answer,
            "metrics": {
                "prompt_tokens": response.prompt_tokens,
                "generated_tokens": 0,
                "total_tokens": response.total_tokens,
                "latency_sec": response.latency_sec,
                "agent_calls": 1,
                "tool_calls": 0,
                "redundant_call": 0,
                "invalid_action": 0,
                "invalid_tool_call_count": 0,
                "premature_termination": 0,
                "late_termination": 0,
            },
        }
