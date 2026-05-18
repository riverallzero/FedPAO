from __future__ import annotations

from typing import TYPE_CHECKING

from src.orchestrators.adapters import ADAPTER_CLASSES, AdapterCheck
from src.orchestrators.mock import MockOrchestrator
from src.orchestrators.native_baselines import NATIVE_ADAPTER_CLASSES
from src.prompts.soft_prompt import SoftPromptProjector

if TYPE_CHECKING:
    from src.models.hf_causal import HFCausalBackend


def adapter_repo_path(config: dict, name: str) -> str | None:
    return (
        config.get("orchestrators", {})
        .get("adapters", {})
        .get(name, {})
        .get("third_party_path")
    )


def adapter_runtime_options(config: dict, name: str) -> dict:
    options = dict(config.get("orchestrators", {}).get(name, {}))
    adapter_options = (
        config.get("orchestrators", {})
        .get("adapters", {})
        .get(name, {})
    )
    options.update(
        {
            key: value
            for key, value in adapter_options.items()
            if key != "third_party_path"
        }
    )
    return options


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
    if kind == "native":
        mas = client_config["mas"]
        if mas not in NATIVE_ADAPTER_CLASSES:
            known = sorted(NATIVE_ADAPTER_CLASSES)
            raise ValueError(f"No native adapter for MAS {mas!r}; expected one of {known}")
        repo_path = adapter_repo_path(config or {}, mas)
        return NATIVE_ADAPTER_CLASSES[mas](
            client_config,
            backend,
            projector,
            max_new_tokens=max_new_tokens,
            repo_path=repo_path,
            **adapter_runtime_options(config or {}, mas),
        )
    if kind not in ADAPTER_CLASSES:
        known = ["mock", "native", *sorted(ADAPTER_CLASSES)]
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
