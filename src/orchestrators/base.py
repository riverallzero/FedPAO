from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class OrchestratorSpec:
    """Static metadata for a baseline orchestrator repo."""

    name: str
    display_name: str
    repo_dir: str
    repo_url: str
    expected_paths: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class OrchestratedPrompt:
    """Prompt and metadata emitted by a FedPAO orchestrator adapter."""

    prompt: str
    metadata: dict[str, Any]


class BaseOrchestrator:
    """Common interface for external MAS baselines.

    The first FedPAO stage intentionally avoids importing external MAS packages
    directly because their dependency pins conflict. Adapters provide a stable
    shape around each cloned repo and can later grow real execution hooks.
    """

    spec: OrchestratorSpec

    def __init__(
        self,
        third_party_dir: Path | str = "third_party",
        **runtime_options: Any,
    ) -> None:
        self.third_party_dir = Path(third_party_dir)
        self.runtime_options = runtime_options

    @property
    def repo_path(self) -> Path:
        return self.third_party_dir / self.spec.repo_dir

    def status(self) -> dict[str, Any]:
        expected = [self.repo_path / relpath for relpath in self.spec.expected_paths]
        return {
            "name": self.spec.name,
            "display_name": self.spec.display_name,
            "repo_url": self.spec.repo_url,
            "repo_path": str(self.repo_path),
            "available": self.repo_path.exists(),
            "expected_paths": [str(path) for path in expected],
            "missing_paths": [str(path) for path in expected if not path.exists()],
        }

    def build_prompt(self, client: dict[str, Any], task_prompt: str) -> OrchestratedPrompt:
        raise NotImplementedError

    def run_inference(
        self,
        client: dict[str, Any],
        example: dict[str, Any],
        task_prompt: str,
        llm_generate: Callable[[str], str],
    ) -> OrchestratedPrompt:
        raise NotImplementedError
