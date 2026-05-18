from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.data.loader import TaskExample
from src.orchestrators.base import OrchestratorResult
from src.prompts.soft_prompt import SoftPromptProjector

if TYPE_CHECKING:
    from src.models.hf_causal import HFCausalBackend


@dataclass(frozen=True)
class AdapterCheck:
    name: str
    repo_path: str
    available: bool
    missing: list[str]
    note: str


class ExternalMasAdapter:
    name = "external"
    default_repo_path = ""
    required_paths: tuple[str, ...] = ()

    def __init__(
        self,
        client_config: dict,
        backend: HFCausalBackend | None,
        projector: SoftPromptProjector | None = None,
        max_new_tokens: int = 32,
        repo_path: str | None = None,
    ):
        self.client_config = client_config
        self.backend = backend
        self.projector = projector
        self.max_new_tokens = max_new_tokens
        self.repo_path = Path(repo_path or self.default_repo_path)

    def check(self) -> AdapterCheck:
        missing = []
        if not self.repo_path.exists():
            missing.append(str(self.repo_path))
        for relative_path in self.required_paths:
            candidate = self.repo_path / relative_path
            if not candidate.exists():
                missing.append(str(candidate))
        return AdapterCheck(
            name=self.name,
            repo_path=str(self.repo_path),
            available=not missing,
            missing=missing,
            note="repo present; native mode is available if dependencies are installed"
            if not missing
            else "required adapter files are missing",
        )

    def run(
        self,
        example: TaskExample,
        prompt_latent: np.ndarray | None = None,
    ) -> OrchestratorResult:
        raise NotImplementedError(
            f"{self.name} adapter is scaffolded but not wired to the external runtime. "
            "Use MockOrchestrator for FedPAO dry runs until dependency isolation and "
            "trajectory extraction are implemented."
        )


class GPTSwarmAdapter(ExternalMasAdapter):
    name = "gptswarm"
    default_repo_path = "third_party/GPTSwarm"
    required_paths = ("swarm",)


class MASRouterAdapter(ExternalMasAdapter):
    name = "masrouter"
    default_repo_path = "third_party/masrouter"
    required_paths = ("MAR",)


class PuppeteerAdapter(ExternalMasAdapter):
    name = "puppeteer"
    default_repo_path = "third_party/ChatDev-puppeteer"
    required_paths = ("puppeteer",)


ADAPTER_CLASSES = {
    GPTSwarmAdapter.name: GPTSwarmAdapter,
    MASRouterAdapter.name: MASRouterAdapter,
    PuppeteerAdapter.name: PuppeteerAdapter,
}
