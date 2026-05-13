from __future__ import annotations

from src.models.hf_causal import HFCausalBackend
from src.orchestrators.adapters import ADAPTER_CLASSES, AdapterCheck
from src.orchestrators.mock import MockOrchestrator
from src.prompts.soft_prompt import SoftPromptProjector


def adapter_repo_path(config: dict, name: str) -> str | None:
    return (
        config.get("orchestrators", {})
        .get("adapters", {})
        .get(name, {})
        .get("third_party_path")
    )


def build_orchestrator(
    kind: str,
    client_config: dict,
    backend: HFCausalBackend,
    projector: SoftPromptProjector | None,
    max_new_tokens: int,
    config: dict | None = None,
):
    if kind == "mock":
        return MockOrchestrator(
            client_config,
            backend,
            projector,
            max_new_tokens=max_new_tokens,
        )
    if kind not in ADAPTER_CLASSES:
        known = ["mock", *sorted(ADAPTER_CLASSES)]
        raise ValueError(f"Unknown orchestrator kind {kind!r}; expected one of {known}")

    repo_path = adapter_repo_path(config or {}, kind)
    return ADAPTER_CLASSES[kind](
        client_config,
        backend,
        projector,
        max_new_tokens=max_new_tokens,
        repo_path=repo_path,
    )


def adapter_statuses(config: dict | None = None) -> list[AdapterCheck]:
    statuses = []
    for name, adapter_class in ADAPTER_CLASSES.items():
        repo_path = adapter_repo_path(config or {}, name)
        adapter = adapter_class(
            client_config={"id": "status", "mas": name},
            backend=None,
            repo_path=repo_path,
        )
        statuses.append(adapter.check())
    return statuses
