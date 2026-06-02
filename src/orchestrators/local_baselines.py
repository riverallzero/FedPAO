from __future__ import annotations

from typing import Any

from src.orchestrators.base import BaseOrchestrator, OrchestratedPrompt, OrchestratorSpec


class LocalBaselineOrchestrator(BaseOrchestrator):
    """Dependency-light simulated adapter for cloned MAS baselines."""

    orchestration_style: str

    def build_prompt(self, client: dict[str, Any], task_prompt: str) -> OrchestratedPrompt:
        status = self.status()
        prompt = (
            f"You are simulating the {self.spec.display_name} multi-agent orchestrator "
            "inside FedPAO.\n"
            f"Orchestration policy: {self.orchestration_style}\n"
            f"Client id: {client['id']}\n"
            f"Dataset: {client['dataset']}\n"
            "Use the local frozen LLM as the shared black-box backend and produce "
            "only the final answer requested by the task.\n\n"
            f"{task_prompt}"
        )
        return OrchestratedPrompt(
            prompt=prompt,
            metadata={
                "adapter_mode": "simulated_local",
                "orchestrator": self.spec.name,
                "orchestrator_display_name": self.spec.display_name,
                "orchestrator_repo_path": str(self.repo_path),
                "orchestrator_repo_available": status["available"],
                "orchestrator_missing_paths": status["missing_paths"],
            },
        )

    def run_inference(self, client, example, task_prompt, llm_generate):
        orchestrated = self.build_prompt(client, task_prompt)
        response = llm_generate(orchestrated.prompt)
        return OrchestratedPrompt(
            prompt=orchestrated.prompt,
            metadata=orchestrated.metadata | {"response": response},
        )


class MASRouterOrchestrator(LocalBaselineOrchestrator):
    spec = OrchestratorSpec(
        name="masrouter",
        display_name="MASRouter",
        repo_dir="masrouter",
        repo_url="https://github.com/yanweiyue/masrouter",
        expected_paths=("Experiments/run_mmlu.py", "Experiments/run_gsm8k.py"),
        summary="Cascaded routing framework for MAS construction and model routing.",
    )
    orchestration_style = (
        "route collaboration mode, roles, and model choices with cost-aware control"
    )


class GPTSwarmOrchestrator(LocalBaselineOrchestrator):
    spec = OrchestratorSpec(
        name="gptswarm",
        display_name="GPT-Swarm",
        repo_dir="GPTSwarm",
        repo_url="https://github.com/metauto-ai/GPTSwarm",
        expected_paths=("swarm", "experiments/run_mmlu.py", "pyproject.toml"),
        summary="Graph-based swarm framework with optimizable agent connections.",
    )
    orchestration_style = (
        "compose graph-style agent reasoning and return the final answer node output"
    )
