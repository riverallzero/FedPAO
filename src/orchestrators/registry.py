from __future__ import annotations

from pathlib import Path

from src.orchestrators.base import BaseOrchestrator
from src.orchestrators.local_baselines import (
    GPTSwarmOrchestrator,
    MASRouterOrchestrator,
)
from src.orchestrators.native_baselines import (
    NativeGPTSwarmOrchestrator,
    NativeMASRouterOrchestrator,
)


ORCHESTRATORS: dict[str, type[BaseOrchestrator]] = {
    "masrouter": MASRouterOrchestrator,
    "gptswarm": GPTSwarmOrchestrator,
}

NATIVE_ORCHESTRATORS: dict[str, type[BaseOrchestrator]] = {
    "masrouter": NativeMASRouterOrchestrator,
    "gptswarm": NativeGPTSwarmOrchestrator,
}

def create_orchestrator(
    name: str,
    third_party_dir: Path | str = "third_party",
    mode: str = "native",
    **runtime_options,
) -> BaseOrchestrator:
    registry = NATIVE_ORCHESTRATORS if mode == "native" else ORCHESTRATORS
    try:
        orchestrator_cls = registry[name]
    except KeyError as exc:
        known = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown orchestrator {name!r} for mode {mode!r}. "
            f"Known orchestrators: {known}"
        ) from exc
    return orchestrator_cls(third_party_dir=third_party_dir, **runtime_options)


def list_orchestrators(third_party_dir: Path | str = "third_party") -> list[dict]:
    return [
        orchestrator_cls(third_party_dir=third_party_dir).status()
        for orchestrator_cls in ORCHESTRATORS.values()
    ]
