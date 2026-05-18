from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np
import yaml

from src.data.loader import TaskExample
from src.orchestrators.base import OrchestratorResult
from src.orchestrators.mock import format_task_prompt
from src.prompts.soft_prompt import SoftPromptProjector

if TYPE_CHECKING:
    from src.models.hf_causal import HFCausalBackend


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
            else:
                role = getattr(message, "role", "user")
                content = getattr(message, "content", str(message))
            parts.append(f"{role}: {content}")
    else:
        parts.append(str(messages))
    return "\n".join(parts)


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
def patched_attr(target: Any, name: str, value: Any) -> Iterator[None]:
    sentinel = object()
    if isinstance(target, type):
        original = target.__dict__.get(name, sentinel)
    else:
        original = getattr(target, name, sentinel)
    setattr(target, name, value)
    try:
        yield
    finally:
        if original is sentinel:
            with contextlib.suppress(AttributeError):
                delattr(target, name)
        else:
            setattr(target, name, original)


@contextlib.contextmanager
def patched_environ(updates: dict[str, str]) -> Iterator[None]:
    sentinel = object()
    original: dict[str, str | object] = {
        key: os.environ.get(key, sentinel) for key in updates
    }
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)


def clear_decorated_singleton(instance_factory: Any) -> None:
    """Clear instances captured by Puppeteer's local Singleton decorator."""
    closure = getattr(instance_factory, "__closure__", None)
    if closure is None:
        return
    for cell in closure:
        with contextlib.suppress(ValueError):
            value = cell.cell_contents
        if isinstance(value, dict):
            value.clear()


def reset_puppeteer_agent_registry(agent_registry: Any) -> None:
    """Drop previously materialized Puppeteer agents before another task run."""
    agent_registry.agents.clear()
    agent_registry.unique_agents.clear()


@contextlib.contextmanager
def projected_yaml_config(
    source_path: Path,
    target_path: Path,
    overrides: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
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
    with open(target_path, "r", encoding="utf-8") as file:
        original_text = file.read()
    with open(source_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})

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


class NativeMasOrchestrator:
    name = "native"
    repo_path_default = ""
    native_entrypoint = ""

    def __init__(
        self,
        client_config: dict,
        backend: HFCausalBackend,
        projector: SoftPromptProjector | None = None,
        max_new_tokens: int = 32,
        repo_path: str | None = None,
        **runtime_options: Any,
    ):
        self.client_config = client_config
        self.backend = backend
        self.projector = projector
        self.max_new_tokens = max_new_tokens
        self.repo_path = Path(repo_path or self.repo_path_default)
        self.runtime_options = runtime_options

    def build_llm_generate(
        self,
        prompt_latent: np.ndarray | None,
        stats: dict[str, float | int],
    ) -> Callable[[str], str]:
        soft_prompt = None
        if prompt_latent is not None and self.projector is not None:
            soft_prompt = self.projector.project(prompt_latent, device=self.backend.device)

        def llm_generate(prompt: str) -> str:
            if soft_prompt is None:
                response = self.backend.generate(prompt, self.max_new_tokens)
            else:
                response = self.backend.generate_with_soft_prompt(
                    prompt,
                    soft_prompt,
                    self.max_new_tokens,
                )
            stats["llm_calls"] += 1
            stats["prompt_tokens"] += response.prompt_tokens
            stats["generated_tokens"] += response.generated_tokens
            stats["total_tokens"] += response.total_tokens
            stats["llm_latency_sec"] += response.latency_sec
            return response.text

        return llm_generate

    def run_native(
        self,
        example: TaskExample,
        task_prompt: str,
        llm_generate: Callable[[str], str],
    ) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def run(
        self,
        example: TaskExample,
        prompt_latent: np.ndarray | None = None,
    ) -> OrchestratorResult:
        task_prompt = format_task_prompt(self.client_config, example)
        stats: dict[str, float | int] = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "total_tokens": 0,
            "llm_latency_sec": 0.0,
        }
        llm_generate = self.build_llm_generate(prompt_latent, stats)
        start = time.perf_counter()
        response, metadata = self.run_native(example, task_prompt, llm_generate)
        latency_sec = round(time.perf_counter() - start, 4)
        total_tokens = int(stats["total_tokens"])
        trajectory = [
            {
                "step_id": 1,
                "selected_component": self.name,
                "capability": "native_mas_inference",
                "tokens": total_tokens,
                "latency_sec": latency_sec,
                "is_redundant": False,
            }
        ]
        metrics = {
            "prompt_tokens": int(stats["prompt_tokens"]),
            "generated_tokens": int(stats["generated_tokens"]),
            "total_tokens": total_tokens,
            "latency_sec": latency_sec,
            "agent_calls": max(1, int(stats["llm_calls"])),
            "tool_calls": 0,
            "redundant_call": 0,
            "invalid_action": 0,
            "invalid_tool_call_count": 0,
            "premature_termination": 0,
            "late_termination": 0,
        }
        return OrchestratorResult(
            trajectory=trajectory,
            final_answer=response,
            metrics=metrics,
            metadata={
                "adapter_mode": "native",
                "orchestrator": self.name,
                "repo_path": str(self.repo_path),
                "native_entrypoint": self.native_entrypoint,
                "llm_calls": int(stats["llm_calls"]),
                "llm_latency_sec": round(float(stats["llm_latency_sec"]), 4),
                **metadata,
            },
        )


class NativePuppeteerOrchestrator(NativeMasOrchestrator):
    name = "puppeteer"
    repo_path_default = "third_party/ChatDev-puppeteer"
    native_entrypoint = "puppeteer.tasks.runner.BenchmarkRunner.run_reasoning"

    def run_native(
        self,
        example: TaskExample,
        task_prompt: str,
        llm_generate: Callable[[str], str],
    ) -> tuple[str, dict[str, Any]]:
        repo = (self.repo_path / "puppeteer").resolve()
        if not repo.exists():
            raise FileNotFoundError(f"Puppeteer repo path does not exist: {repo}")

        reward_model_path = self.runtime_options.get("reward_model_path")
        global_config_path = Path(
            self.runtime_options.get(
                "global_config",
                "configs/orchestrators/puppeteer_global.yaml",
            )
        ).resolve()
        policy_config_path = Path(
            self.runtime_options.get(
                "policy_config",
                "configs/orchestrators/puppeteer_policy.json",
            )
        ).resolve()

        with import_context(repo, cwd=repo):
            import openai

            class _ImportOnlyOpenAI:
                def __init__(self, *args, **kwargs):
                    pass

            disable_reward_model = bool(
                self.runtime_options.get("disable_reward_model", False)
            )
            policy_overrides = {"device": {"type": "cpu"}} if disable_reward_model else None

            with (
                projected_yaml_config(
                    global_config_path,
                    repo / "config" / "global.yaml",
                    {"model_weight_path": reward_model_path},
                ) as global_config,
                projected_json_config(
                    policy_config_path,
                    repo / "config" / "policy.json",
                    policy_overrides,
                ) as policy_config,
                patched_attr(openai, "OpenAI", _ImportOnlyOpenAI),
            ):
                try:
                    from inference.policy import REINFORCE_continuous as reinforce_module

                    ContinuousREINFORCE = reinforce_module.ContinuousREINFORCE
                    from agent.register.register import agent_global_registry
                    from inference.reasoning.reasoning import GraphReasoning
                    from model.query_manager import query_manager
                    from tasks.runner import BenchmarkRunner
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "Puppeteer native execution requires third_party dependencies. "
                        f"Missing module: {exc.name}"
                    ) from exc

                def local_query(model_key, messages, system_prompt=None):
                    prompt = messages_to_prompt(messages, system_prompt)
                    response = llm_generate(prompt)
                    return response, max(1, len(prompt.split()) + len(response.split()))

                class _ZeroRewardModelTokenRepresentation:
                    dim = 8192

                    def __call__(self, messages):
                        import torch

                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        return torch.zeros((1, self.dim), device=device), 0.0

                reward_context = (
                    patched_attr(
                        reinforce_module,
                        "RewardModelTokenRepresentation",
                        _ZeroRewardModelTokenRepresentation,
                    )
                    if disable_reward_model
                    else contextlib.nullcontext()
                )
                original_aggregate_answers = GraphReasoning.aggregate_answers

                def safe_aggregate_answers(self, global_info, answers, query_func=None):
                    if not answers:
                        return ""
                    return original_aggregate_answers(self, global_info, answers, query_func)

                with (
                    patched_attr(query_manager, "query", local_query),
                    patched_attr(
                        ContinuousREINFORCE,
                        "load_model",
                        lambda self, path, strict=True: False,
                    ),
                    patched_attr(
                        ContinuousREINFORCE,
                        "save_model",
                        lambda self, path=None, tag=None: None,
                    ),
                    patched_attr(reinforce_module.atexit, "register", lambda *args, **kwargs: None),
                    patched_attr(GraphReasoning, "aggregate_answers", safe_aggregate_answers),
                    reward_context,
                ):
                    runner = BenchmarkRunner("personas/personas.jsonl", global_config)
                    if example.dataset == "mmlu_pro":
                        task = {
                            "type": "MMLU-Pro",
                            "Question": mmlu_question(example.raw),
                            "Answer": example.answer,
                            "id": example.raw.get("question_id", example.task_id),
                        }
                    elif example.dataset == "gsm_hard":
                        task = {
                            "type": "GSM-Hard",
                            "Question": "You need to write python program to solve math problems:\n"
                            + example.question,
                            "Answer": example.answer,
                            "id": example.task_id,
                        }
                    else:
                        raise ValueError(f"Unsupported dataset for Puppeteer: {example.dataset}")
                    reset_puppeteer_agent_registry(agent_global_registry)
                    clear_decorated_singleton(reinforce_module.MLP_PolicyNetwork)
                    clear_decorated_singleton(ContinuousREINFORCE)
                    response = runner.run_reasoning(task)

        return str(response), {
            "puppeteer_reward_model_path": reward_model_path,
            "puppeteer_global_config": str(global_config_path),
            "puppeteer_policy_config": str(policy_config_path),
            "puppeteer_policy_training": policy_config.get("training", {}).get("training"),
            "puppeteer_policy_loading": policy_config.get("training", {}).get("loading"),
            "puppeteer_reward_model_disabled": bool(
                self.runtime_options.get("disable_reward_model", False)
            ),
        }


class NativeGPTSwarmOrchestrator(NativeMasOrchestrator):
    name = "gptswarm"
    repo_path_default = "third_party/GPTSwarm"
    native_entrypoint = "swarm.graph.swarm.Swarm.run"

    def run_native(
        self,
        example: TaskExample,
        task_prompt: str,
        llm_generate: Callable[[str], str],
    ) -> tuple[str, dict[str, Any]]:
        if not self.repo_path.exists():
            raise FileNotFoundError(f"GPTSwarm repo path does not exist: {self.repo_path}")

        with import_context(self.repo_path):
            try:
                from swarm.environment.prompt.common import get_combine_materials
                from swarm.environment.prompt.prompt_set import PromptSet
                from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
                from swarm.graph.swarm import Swarm
                from swarm.llm.llm import LLM
                from swarm.llm.llm_registry import LLMRegistry
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "GPT-Swarm native execution requires third_party dependencies. "
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

            if "gsm_hard" not in PromptSetRegistry.keys():

                @PromptSetRegistry.register("gsm_hard")
                class GSMHardPromptSet(PromptSet):
                    @staticmethod
                    def get_role():
                        return "careful mathematical reasoning assistant"

                    @staticmethod
                    def get_constraint():
                        return "Solve the math problem. Return only the final numeric answer."

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
                        return f"Problem: {question}\nAnswer: {answer}"

                    @staticmethod
                    def get_combine_materials(materials):
                        return get_combine_materials(materials)

            if example.dataset == "mmlu_pro":
                domain = "mmlu"
                task = mmlu_question(example.raw)
            elif example.dataset == "gsm_hard":
                domain = "gsm_hard"
                task = gsm_question(example.raw)
            else:
                raise ValueError(f"Unsupported dataset for GPT-Swarm: {example.dataset}")

            with patched_attr(
                LLMRegistry,
                "get",
                classmethod(lambda cls, model_name=None: FedPAOLocalLLM(model_name)),
            ):
                swarm = Swarm(
                    ["IO", "COT", "IO"],
                    domain,
                    model_name="fedpao_local",
                    edge_optimize=False,
                )
                response = swarm.run({"task": task})

        return str(response), {
            "swarm_domain": domain,
            "swarm_agents": ["IO", "COT", "IO"],
        }


class NativeMASRouterOrchestrator(NativeMasOrchestrator):
    name = "masrouter"
    repo_path_default = "third_party/masrouter"
    native_entrypoint = "MAR.MasRouter.mas_router.MasRouter.forward"

    def run_native(
        self,
        example: TaskExample,
        task_prompt: str,
        llm_generate: Callable[[str], str],
    ) -> tuple[str, dict[str, Any]]:
        repo = self.repo_path.resolve()
        if not repo.exists():
            raise FileNotFoundError(f"MASRouter repo path does not exist: {repo}")

        masrouter_cache = repo / "data" / "cache" / "huggingface"
        env_updates = {
            "HF_HOME": str(masrouter_cache),
            "TRANSFORMERS_CACHE": str(masrouter_cache / "hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }

        with patched_environ(env_updates), import_context(repo, cwd=repo):
            try:
                import torch

                bootstrap_masrouter_imports(repo)
                from MAR.Agent.reasoning_profile import reasoning_profile
                from MAR.LLM.llm import LLM
                from MAR.LLM import llm_embedding
                from MAR.LLM.llm_profile import llm_profile
                from MAR.LLM.llm_registry import LLMRegistry
                from MAR.MasRouter.mas_router import MasRouter
                from MAR.Prompts.tasks_profile import tasks_profile
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "MASRouter native execution requires third_party dependencies. "
                    f"Missing module: {exc.name}"
                ) from exc

            class FedPAOLocalLLM(LLM):
                def __init__(self, model_name=None):
                    self.model_name = model_name or "fedpao_local"

                async def agen(self, messages, max_tokens=None, temperature=None, num_comps=None):
                    return self.gen(messages, max_tokens, temperature, num_comps)

                def gen(self, messages, max_tokens=None, temperature=None, num_comps=None):
                    return llm_generate(messages_to_prompt(messages))

            if example.dataset == "mmlu_pro":
                query = mmlu_question(example.raw)
                task_label = 1
                prompt_file = "MAR/Roles/FinalNode/mmlu.json"
            elif example.dataset == "gsm_hard":
                query = gsm_question(example.raw)
                task_label = 0
                prompt_file = "MAR/Roles/FinalNode/gsm8k.json"
            else:
                raise ValueError(f"Unsupported dataset for MASRouter: {example.dataset}")

            with patched_attr(
                LLMRegistry,
                "get",
                classmethod(lambda cls, model_name=None: FedPAOLocalLLM(model_name)),
            ), patched_attr(
                llm_embedding,
                "SentenceTransformer",
                self._local_masrouter_sentence_transformer(llm_embedding, repo),
            ):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                router = MasRouter(max_agent=3, device=device).to(device).eval()
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

        return str(response), {
            "cost": float(costs[0]) if costs else None,
            "agents_num": float(agents_num[0].item()) if len(agents_num) else None,
        }

    @staticmethod
    def _local_masrouter_sentence_transformer(llm_embedding_module: Any, repo: Path):
        original = llm_embedding_module.SentenceTransformer
        snapshots_dir = (
            repo
            / "data"
            / "cache"
            / "huggingface"
            / "hub"
            / "models--sentence-transformers--all-MiniLM-L6-v2"
            / "snapshots"
        )
        snapshots = (
            sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
            if snapshots_dir.exists()
            else []
        )
        local_model = snapshots[-1] if snapshots else None

        def load(model_name_or_path: str, *args: Any, **kwargs: Any):
            if (
                model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2"
                and local_model is not None
            ):
                kwargs.setdefault("local_files_only", True)
                return original(str(local_model), *args, **kwargs)
            return original(model_name_or_path, *args, **kwargs)

        return load


NATIVE_ADAPTER_CLASSES = {
    "puppeteer": NativePuppeteerOrchestrator,
    "masrouter": NativeMASRouterOrchestrator,
    "gptswarm": NativeGPTSwarmOrchestrator,
}
