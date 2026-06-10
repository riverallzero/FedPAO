from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import yaml

from src.orchestrators.base import OrchestratedPrompt
from src.orchestrators.local_baselines import (
    GPTSwarmOrchestrator,
    MASRouterOrchestrator,
)


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        elif value is not None:
            base[key] = value
    return base


@contextlib.contextmanager
def import_context(path: Path, cwd: Path | None = None) -> Iterator[None]:
    old_cwd = Path.cwd()
    path_str = str(path.resolve())
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    if cwd is not None:
        os.chdir(cwd)
    try:
        yield
    finally:
        if cwd is not None:
            os.chdir(old_cwd)
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path_str)


def rough_token_count(text: str) -> int:
    return max(1, len(text.split()))


def messages_to_prompt(messages: Any, system_prompt: str | None = None) -> str:
    parts = []
    if system_prompt:
        parts.append(f"system: {system_prompt}")
    if isinstance(messages, str):
        parts.append(f"user: {messages}")
    elif isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                role = getattr(message, "role", "user")
                content = getattr(message, "content", str(message))
                parts.append(f"{role}: {content}")
    else:
        parts.append(str(messages))
    return "\n".join(parts)


def native_metadata(orchestrator, entrypoint: str) -> dict[str, Any]:
    status = orchestrator.status()
    return {
        "adapter_mode": "native",
        "orchestrator": orchestrator.spec.name,
        "orchestrator_display_name": orchestrator.spec.display_name,
        "orchestrator_repo_path": status["repo_path"],
        "orchestrator_repo_available": status["available"],
        "orchestrator_missing_paths": status["missing_paths"],
        "native_entrypoint": entrypoint,
    }


def mmlu_question(example: dict[str, Any]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = " ".join(
        f"{letters[idx]}: {option}" for idx, option in enumerate(example["options"])
    )
    category = example.get("category", "general knowledge")
    return (
        f"The following are multiple choice questions about {category}.\n"
        f"{example['question']}\n"
        f"{options}\n"
        "Return only the option letter."
    )


def gsm_question(example: dict[str, Any]) -> str:
    return (
        "Solve the math problem. Return only the final numeric answer.\n"
        f"{example['input']}"
    )


def load_module_from_path(module_name: str, path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_masrouter_imports(repo: Path) -> None:
    """Preload MASRouter modules in an order that avoids package circular imports."""

    graph_pkg = sys.modules.get("MAR.Graph")
    if graph_pkg is None:
        graph_pkg = types.ModuleType("MAR.Graph")
        graph_pkg.__path__ = [str(repo / "MAR" / "Graph")]
        sys.modules["MAR.Graph"] = graph_pkg

    node_module = load_module_from_path(
        "MAR.Graph.node",
        repo / "MAR" / "Graph" / "node.py",
    )
    graph_pkg.Node = node_module.Node

    load_module_from_path(
        "MAR.Agent.agent_registry",
        repo / "MAR" / "Agent" / "agent_registry.py",
    )
    load_module_from_path(
        "MAR.Agent.agent",
        repo / "MAR" / "Agent" / "agent.py",
    )
    graph_module = load_module_from_path(
        "MAR.Graph.graph",
        repo / "MAR" / "Graph" / "graph.py",
    )
    graph_pkg.Graph = graph_module.Graph


@contextlib.contextmanager
def projected_yaml_config(
    source_path: Path,
    target_path: Path,
    overrides: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Project a FedPAO-owned YAML config into a third_party runtime path."""

    with open(target_path, "r", encoding="utf-8") as file:
        original_text = file.read()
    with open(source_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target_path.parent,
        delete=False,
    ) as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
        temp_name = file.name
    os.replace(temp_name, target_path)
    try:
        yield config
    finally:
        with open(target_path, "w", encoding="utf-8") as file:
            file.write(original_text)


@contextlib.contextmanager
def projected_json_config(
    source_path: Path,
    target_path: Path,
    overrides: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Project a FedPAO-owned JSON config into a third_party runtime path."""

    with open(target_path, "r", encoding="utf-8") as file:
        original_text = file.read()
    with open(source_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if overrides:
        deep_update(config, overrides)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target_path.parent,
        delete=False,
    ) as file:
        json.dump(config, file, indent=4)
        file.write("\n")
        temp_name = file.name
    os.replace(temp_name, target_path)
    try:
        yield config
    finally:
        with open(target_path, "w", encoding="utf-8") as file:
            file.write(original_text)


class NativeGPTSwarmOrchestrator(GPTSwarmOrchestrator):
    """Run GPT-Swarm graph execution with a FedPAO local LLM registry entry."""

    def run_inference(
        self,
        client: dict[str, Any],
        example: dict[str, Any],
        task_prompt: str,
        llm_generate: Callable[[str], str],
    ) -> OrchestratedPrompt:
        repo = self.repo_path
        if not repo.exists():
            raise FileNotFoundError(f"GPTSwarm repo path does not exist: {repo}")

        with import_context(repo):
            try:
                from swarm.graph.swarm import Swarm
                from swarm.environment.prompt.common import get_combine_materials
                from swarm.environment.prompt.prompt_set import PromptSet
                from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
                from swarm.llm.llm import LLM
                from swarm.llm.llm_registry import LLMRegistry
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "GPT-Swarm native execution requires its third_party dependencies. "
                    f"Missing module: {exc.name}"
                ) from exc

            class FedPAOLocalLLM(LLM):
                def __init__(self, model_name=None):
                    self.model_name = model_name or "fedpao_local"

                async def agen(self, messages, max_tokens=None, temperature=None, num_comps=None):
                    return self.gen(messages, max_tokens, temperature, num_comps)

                def gen(self, messages, max_tokens=None, temperature=None, num_comps=None):
                    return llm_generate(messages_to_prompt(messages))

            if "fedpao_local" not in LLMRegistry.keys():
                LLMRegistry.register("fedpao_local")(FedPAOLocalLLM)
            LLMRegistry.get = classmethod(
                lambda cls, model_name=None: FedPAOLocalLLM(model_name)
            )

            if "gsm_hard" not in PromptSetRegistry.keys():
                @PromptSetRegistry.register("gsm_hard")
                class GSMHardPromptSet(PromptSet):
                    @staticmethod
                    def get_role():
                        return "careful mathematical reasoning assistant"

                    @staticmethod
                    def get_constraint():
                        return (
                            "Solve the given math problem. Return only the final "
                            "numeric answer. Do not include units, explanations, "
                            "or formatting unless the problem explicitly requires it."
                        )

                    @staticmethod
                    def get_format():
                        return "final numeric answer"

                    @staticmethod
                    def get_answer_prompt(question):
                        return str(question)

                    @staticmethod
                    def get_adversarial_answer_prompt(question):
                        return f"Give an incorrect numeric answer to: {question}"

                    @staticmethod
                    def get_query_prompt(question):
                        return f"Identify the arithmetic steps needed to solve: {question}"

                    @staticmethod
                    def get_file_analysis_prompt(query, file):
                        return f"{query}\n\n{file}"

                    @staticmethod
                    def get_websearch_prompt(query):
                        return str(query)

                    @staticmethod
                    def get_distill_websearch_prompt(query, results):
                        return f"{query}\n\n{results}"

                    @staticmethod
                    def get_reflect_prompt(question, answer):
                        return (
                            "Check whether this numeric answer follows from the math "
                            f"problem.\nProblem: {question}\nAnswer: {answer}"
                        )

                    @staticmethod
                    def get_combine_materials(materials):
                        return get_combine_materials(materials)

            if client["dataset"] == "mmlu_pro":
                domain = "mmlu"
                task = mmlu_question(example)
            elif client["dataset"] == "gsm_hard":
                domain = "gsm_hard"
                task = gsm_question(example)
            else:
                raise ValueError(f"Unsupported dataset for GPT-Swarm: {client['dataset']}")

            round0_checkpoint_path = self.runtime_options.get("round0_checkpoint_path")
            swarm = Swarm(
                ["IO", "COT", "IO"],
                domain,
                model_name="fedpao_local",
                edge_optimize=bool(round0_checkpoint_path),
            )
            realized_graph = None
            loaded_edges = False
            if round0_checkpoint_path:
                try:
                    checkpoint = torch.load(
                        round0_checkpoint_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                except TypeError:
                    checkpoint = torch.load(round0_checkpoint_path, map_location="cpu")
                state_dict = checkpoint.get("connection_dist_state_dict")
                if state_dict:
                    swarm.connection_dist.load_state_dict(state_dict, strict=False)
                    edge_probs = torch.sigmoid(swarm.connection_dist.edge_logits)
                    realized_graph = swarm.connection_dist.realize_mask(
                        swarm.composite_graph,
                        edge_probs > 0.5,
                    )
                    loaded_edges = True
            response = swarm.run({"task": task}, realized_graph=realized_graph)

        return OrchestratedPrompt(
            prompt=task_prompt,
            metadata=native_metadata(
                self,
                "swarm.graph.swarm.Swarm.run",
            )
            | {
                "swarm_domain": domain,
                "swarm_agents": ["IO", "COT", "IO"],
                "round0_checkpoint_path": str(round0_checkpoint_path) if round0_checkpoint_path else None,
                "gptswarm_loaded_edges": loaded_edges,
                "response": str(response),
            },
        )


class NativeMASRouterOrchestrator(MASRouterOrchestrator):
    """Run MASRouter routing and graph execution with FedPAO local LLM patched in."""

    def run_inference(
        self,
        client: dict[str, Any],
        example: dict[str, Any],
        task_prompt: str,
        llm_generate: Callable[[str], str],
    ) -> OrchestratedPrompt:
        repo = self.repo_path.resolve()
        if not repo.exists():
            raise FileNotFoundError(f"MASRouter repo path does not exist: {repo}")

        with import_context(repo, cwd=repo):
            try:
                import torch
                bootstrap_masrouter_imports(repo)
                agent_module = importlib.import_module("MAR.Agent.agent")
                post_process_module = importlib.import_module("MAR.Prompts.post_process")
                from MAR.Agent.reasoning_profile import reasoning_profile
                from MAR.LLM.llm import LLM
                from MAR.LLM.llm_profile import llm_profile
                from MAR.LLM.llm_registry import LLMRegistry
                from MAR.MasRouter.mas_router import MasRouter
                from MAR.Prompts.tasks_profile import tasks_profile
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "MASRouter native execution requires its third_party dependencies. "
                    f"Missing module: {exc.name}"
                ) from exc

            class FedPAOLocalLLM(LLM):
                def __init__(self, model_name=None):
                    self.model_name = model_name or "fedpao_local"

                async def agen(self, messages, max_tokens=None, temperature=None, num_comps=None):
                    return self.gen(messages, max_tokens, temperature, num_comps)

                def gen(self, messages, max_tokens=None, temperature=None, num_comps=None):
                    return llm_generate(messages_to_prompt(messages))

            LLMRegistry.get = classmethod(
                lambda cls, model_name=None: FedPAOLocalLLM(model_name)
            )
            original_post_process = post_process_module.post_process

            def safe_post_process(raw_inputs, output, post_method):
                try:
                    return original_post_process(raw_inputs, output, post_method)
                except Exception as exc:
                    return (
                        f"{output}\n\n"
                        "[FedPAO warning: MASRouter post_process failed; returning raw "
                        f"LLM output. reason={exc}]"
                    )

            post_process_module.post_process = safe_post_process
            agent_module.post_process = safe_post_process

            if client["dataset"] == "mmlu_pro":
                query = mmlu_question(example)
                task_label = 1
                prompt_file = "MAR/Roles/FinalNode/mmlu.json"
            elif client["dataset"] == "gsm_hard":
                query = gsm_question(example)
                task_label = 0
                prompt_file = "MAR/Roles/FinalNode/gsm8k.json"
            else:
                raise ValueError(f"Unsupported dataset for MASRouter: {client['dataset']}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            router = MasRouter(max_agent=3, device=device).to(device).eval()
            round0_checkpoint_path = self.runtime_options.get("round0_checkpoint_path")
            masrouter_loaded_state = False
            if round0_checkpoint_path:
                try:
                    checkpoint = torch.load(
                        round0_checkpoint_path,
                        map_location=device,
                        weights_only=False,
                    )
                except TypeError:
                    checkpoint = torch.load(round0_checkpoint_path, map_location=device)
                state_dict = checkpoint.get("router_state_dict", checkpoint)
                router.load_state_dict(state_dict, strict=False)
                masrouter_loaded_state = True
            with torch.no_grad():
                results, costs, _, _, _, agents_num = router.forward(
                    [query],
                    tasks_profile,
                    llm_profile,
                    reasoning_profile,
                    [task_label],
                    prompt_file=prompt_file,
                )
            response = results[0]

        return OrchestratedPrompt(
            prompt=task_prompt,
            metadata=native_metadata(
                self,
                "MAR.MasRouter.mas_router.MasRouter.forward",
            )
            | {
                "cost": float(costs[0]) if costs else None,
                "agents_num": float(agents_num[0].item()) if len(agents_num) else None,
                "round0_checkpoint_path": str(round0_checkpoint_path) if round0_checkpoint_path else None,
                "masrouter_loaded_state": masrouter_loaded_state,
                "response": str(response),
            },
        )
