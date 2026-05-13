from __future__ import annotations

import numpy as np

from src.data.loader import TaskExample
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
