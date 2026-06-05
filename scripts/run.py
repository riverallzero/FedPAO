from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import importlib
import io
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_client_inference import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    SUPPORTED_MAS,
    generate,
    is_correct,
    load_backend,
    make_prompt,
    normalize_prediction,
)
from src.orchestrators import create_orchestrator  # noqa: E402
from src.orchestrators.native_baselines import (  # noqa: E402
    bootstrap_masrouter_imports,
    gsm_question,
    import_context,
    messages_to_prompt,
    mmlu_question,
    projected_json_config,
    projected_yaml_config,
)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "run"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def selected_clients(config: dict[str, Any], mas: list[str] | None) -> list[dict[str, Any]]:
    selected_mas = set(mas) if mas else None
    clients = [
        client
        for client in config["clients"]
        if selected_mas is None or client["mas"] in selected_mas
    ]
    if not clients:
        raise ValueError(f"No clients matched --mas {mas}")
    return clients


def dataset_size(client: dict[str, Any], limit: int | None) -> tuple[str, int]:
    try:
        dataset = load_dataset(client["dataset_name"])
    except Exception as exc:
        fallback_size = limit if limit is not None else 0
        print(
            f"warning: could not inspect {client['dataset_name']} for {client['id']}; "
            f"using num_examples={fallback_size}. reason={exc}"
        )
        return "unavailable", fallback_size
    split = "train" if "train" in dataset else list(dataset.keys())[0]
    size = len(dataset[split])
    if limit is not None:
        size = min(size, limit)
    return split, size


def load_client_examples(
    client: dict[str, Any],
    split_preference: str,
    limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    dataset = load_dataset(client["dataset_name"])
    if split_preference in dataset:
        split = split_preference
    elif split_preference == "train" and "validation" in dataset:
        split = "validation"
    else:
        split = list(dataset.keys())[0]
    rows = dataset[split].select(range(min(limit, len(dataset[split]))))
    return split, [dict(row) for row in rows]


def full_benchmark_dataset(client: dict[str, Any]):
    dataset = load_dataset(client["dataset_name"])
    splits = [dataset[name] for name in dataset.keys()]
    if len(splits) == 1:
        return splits[0]
    return concatenate_datasets(splits)


def split_client_examples(
    client: dict[str, Any],
    train_fraction: float,
    seed: int,
    sample_limit: int | None = None,
    train_limit: int | None = None,
    valid_limit: int | None = None,
) -> tuple[str, list[dict[str, Any]], str, list[dict[str, Any]]]:
    dataset = full_benchmark_dataset(client)
    indices = list(range(len(dataset)))
    split_key = client.get("dataset_name") or client.get("dataset") or client["id"]
    split_seed = seed + sum(ord(char) for char in split_key)
    random.Random(split_seed).shuffle(indices)
    if sample_limit is not None:
        indices = indices[:sample_limit]
    train_size = int(len(indices) * train_fraction)
    train_size = min(max(train_size, 1), max(len(indices) - 1, 1)) if len(indices) > 1 else len(indices)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    if train_limit is not None:
        train_indices = train_indices[:train_limit]
    if valid_limit is not None:
        valid_indices = valid_indices[:valid_limit]
    train_examples = [dict(row) for row in dataset.select(train_indices)]
    valid_examples = [dict(row) for row in dataset.select(valid_indices)]
    sample_label = "full" if sample_limit is None else f"sample{len(indices)}"
    train_pct = int(train_fraction * 100)
    valid_pct = 100 - train_pct
    return f"{sample_label}_train_{train_pct}pct", train_examples, f"{sample_label}_valid_{valid_pct}pct", valid_examples


def split_client_examples_with_trajectory(
    client: dict[str, Any],
    train_fraction: float,
    seed: int,
    sample_limit: int | None = None,
    train_limit: int | None = None,
    valid_limit: int | None = None,
    trajectory_limit: int | None = None,
) -> tuple[
    str,
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
]:
    dataset = full_benchmark_dataset(client)
    indices = list(range(len(dataset)))
    split_key = client.get("dataset_name") or client.get("dataset") or client["id"]
    split_seed = seed + sum(ord(char) for char in split_key)
    random.Random(split_seed).shuffle(indices)

    train_valid_indices = indices[:sample_limit] if sample_limit is not None else indices
    trajectory_indices = indices[sample_limit:] if sample_limit is not None else []
    train_size = int(len(train_valid_indices) * train_fraction)
    train_size = (
        min(max(train_size, 1), max(len(train_valid_indices) - 1, 1))
        if len(train_valid_indices) > 1
        else len(train_valid_indices)
    )
    train_indices = train_valid_indices[:train_size]
    valid_indices = train_valid_indices[train_size:]
    if train_limit is not None:
        train_indices = train_indices[:train_limit]
    if valid_limit is not None:
        valid_indices = valid_indices[:valid_limit]

    if not trajectory_indices:
        used_indices = set(train_indices) | set(valid_indices)
        trajectory_indices = [index for index in indices if index not in used_indices]
    if trajectory_limit is not None:
        trajectory_indices = trajectory_indices[:trajectory_limit]

    train_examples = [dict(row) for row in dataset.select(train_indices)]
    valid_examples = [dict(row) for row in dataset.select(valid_indices)]
    trajectory_examples = [dict(row) for row in dataset.select(trajectory_indices)]
    sample_label = (
        "full" if sample_limit is None else f"sample{len(train_valid_indices)}"
    )
    train_pct = int(train_fraction * 100)
    valid_pct = 100 - train_pct
    return (
        f"{sample_label}_train_{train_pct}pct",
        train_examples,
        f"{sample_label}_valid_{valid_pct}pct",
        valid_examples,
        f"{sample_label}_heldout",
        trajectory_examples,
    )


def validation_split_name(dataset, train_split: str | None = None) -> str:
    for split in ("validation", "test", "dev"):
        if split in dataset and split != train_split:
            return split
    if train_split in dataset:
        return train_split
    return list(dataset.keys())[0]


def latest_file(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern), key=lambda item: item.stat().st_mtime)
    return files[-1] if files else None


def numeric_equal(left: str, right: str) -> bool:
    try:
        return float(left) == float(right)
    except ValueError:
        return left.strip() == right.strip()


def safe_gsm_prediction(text: str, parser=None) -> str:
    if parser is not None:
        try:
            return str(parser(text))
        except Exception:
            pass
    return normalize_prediction("gsm_hard", text)


def reset_imported_modules(prefixes: tuple[str, ...]) -> None:
    root_names = tuple(prefix.rstrip(".") for prefix in prefixes)
    for module_name in list(sys.modules):
        if module_name in root_names or module_name.startswith(prefixes):
            sys.modules.pop(module_name, None)


def register_gptswarm_local_backend(llm_generate) -> None:
    from swarm.environment.prompt.common import get_combine_materials
    from swarm.environment.prompt.prompt_set import PromptSet
    from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
    from swarm.llm.llm import LLM
    from swarm.llm.llm_registry import LLMRegistry

    class FedPAOLocalLLM(LLM):
        def __init__(self, model_name=None):
            self.model_name = model_name or "fedpao_local"

        async def agen(self, messages, max_tokens=None, temperature=None, num_comps=None):
            return self.gen(messages, max_tokens, temperature, num_comps)

        def gen(self, messages, max_tokens=None, temperature=None, num_comps=None):
            return llm_generate(messages_to_prompt(messages))

    if "fedpao_local" not in LLMRegistry.keys():
        LLMRegistry.register("fedpao_local")(FedPAOLocalLLM)
    LLMRegistry.get = classmethod(lambda cls, model_name=None: FedPAOLocalLLM(model_name))

    if "gsm_hard" not in PromptSetRegistry.keys():
        @PromptSetRegistry.register("gsm_hard")
        class GSMHardPromptSet(PromptSet):
            @staticmethod
            def get_role():
                return "careful mathematical reasoning assistant"

            @staticmethod
            def get_constraint():
                return "Solve the math problem and return only the final numeric answer."

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
                return f"Problem: {question}\nAnswer: {answer}\nCheck the answer."

            @staticmethod
            def get_combine_materials(materials):
                return get_combine_materials(materials)


async def run_gptswarm_batch(swarm, domain: str, client: dict[str, Any], batch):
    tasks = []
    log_probs = []
    answers = []
    for example in batch:
        graph, log_prob = swarm.connection_dist.realize(swarm.composite_graph)
        if client["dataset"] == "mmlu_pro":
            task = mmlu_question(example)
            answer = str(example["answer"]).strip()
        else:
            task = gsm_question(example)
            answer = str(example["target"]).strip()
        tasks.append(swarm.arun({"task": task}, graph))
        log_probs.append(log_prob)
        answers.append(answer)
    raw_answers = await asyncio.gather(*tasks)
    utilities = []
    losses = []
    for raw_answer, log_prob, answer in zip(raw_answers, log_probs, answers):
        prediction = normalize_prediction(client["dataset"], str(raw_answer))
        utility = float(is_correct(client["dataset"], prediction, answer))
        utilities.append(utility)
        losses.append(-log_prob * utility)
    return torch.stack(losses).mean(), utilities


def epoch_batches(examples: list[dict[str, Any]], batch_size: int, epochs: int):
    for epoch in range(epochs):
        for start in range(0, len(examples), batch_size):
            yield epoch, examples[start : start + batch_size]


def train_gptswarm_native(
    args: argparse.Namespace,
    config: dict[str, Any],
    client: dict[str, Any],
    examples: list[dict[str, Any]],
    checkpoint_dir: Path,
    llm_generate,
) -> tuple[Path, dict[str, Any]]:
    repo = (Path(args.third_party_dir) / "GPTSwarm").resolve()
    if not repo.exists():
        raise FileNotFoundError(f"GPTSwarm repo path does not exist: {repo}")

    with import_context(repo):
        from swarm.graph.swarm import Swarm

        register_gptswarm_local_backend(llm_generate)
        domain = "mmlu" if client["dataset"] == "mmlu_pro" else "gsm_hard"
        swarm = Swarm(
            ["IO", "COT", "IO"],
            domain,
            model_name="fedpao_local",
            edge_optimize=True,
        )
        optimizer = torch.optim.Adam(
            swarm.connection_dist.parameters(),
            lr=float(config.get("optimizer", {}).get("learning_rate", 0.1)),
        )
        epochs = int(config.get("optimizer", {}).get("local_iterations", 5))
        batch_size = max(1, min(len(examples), int(config.get("optimizer", {}).get("local_batch_size", 1))))
        num_steps = 0
        for step, (epoch, batch) in enumerate(epoch_batches(examples, batch_size, epochs)):
            optimizer.zero_grad()
            loss, utilities = asyncio.run(run_gptswarm_batch(swarm, domain, client, batch))
            loss.backward()
            optimizer.step()
            num_steps = step + 1

        edge_probs = torch.sigmoid(swarm.connection_dist.edge_logits).detach().cpu()
        checkpoint = {
            "format_version": 1,
            "kind": "gptswarm_edges",
            "round": 0,
            "client_id": client["id"],
            "mas": client["mas"],
            "dataset": client["dataset"],
            "domain": domain,
            "agent_names": ["IO", "COT", "IO"],
            "connection_dist_state_dict": swarm.connection_dist.state_dict(),
            "edge_probs": edge_probs,
        }
    checkpoint_path = checkpoint_dir / "gptswarm_edges_round0.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, {
        "trainer": "GPTSwarm edge_optimizer",
        "num_edges": int(edge_probs.numel()),
        "num_examples": len(examples),
        "epochs": epochs,
        "batch_size": batch_size,
        "num_train_steps": num_steps,
    }


def patch_masrouter_local_backend(llm_generate):
    from MAR.LLM.llm import LLM
    from MAR.LLM.llm_registry import LLMRegistry
    agent_module = importlib.import_module("MAR.Agent.agent")
    post_process_module = importlib.import_module("MAR.Prompts.post_process")

    class FedPAOLocalLLM(LLM):
        def __init__(self, model_name=None):
            self.model_name = model_name or "fedpao_local"

        async def agen(self, messages, max_tokens=None, temperature=None, num_comps=None):
            return self.gen(messages, max_tokens, temperature, num_comps)

        def gen(self, messages, max_tokens=None, temperature=None, num_comps=None):
            return llm_generate(messages_to_prompt(messages))

    LLMRegistry.get = classmethod(lambda cls, model_name=None: FedPAOLocalLLM(model_name))
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


def masrouter_batch(client: dict[str, Any], examples: list[dict[str, Any]]) -> tuple[list[str], list[str], list[int], str]:
    if client["dataset"] == "mmlu_pro":
        return (
            [mmlu_question(example) for example in examples],
            [str(example["answer"]).strip() for example in examples],
            [1 for _ in examples],
            "MAR/Roles/FinalNode/mmlu.json",
        )
    return (
        [gsm_question(example) for example in examples],
        [str(example["target"]).strip() for example in examples],
        [0 for _ in examples],
        "MAR/Roles/FinalNode/gsm8k.json",
    )


def train_masrouter_native(
    args: argparse.Namespace,
    config: dict[str, Any],
    client: dict[str, Any],
    examples: list[dict[str, Any]],
    checkpoint_dir: Path,
    llm_generate,
) -> tuple[Path, dict[str, Any]]:
    repo = (Path(args.third_party_dir) / "masrouter").resolve()
    if not repo.exists():
        raise FileNotFoundError(f"MASRouter repo path does not exist: {repo}")

    with import_context(repo, cwd=repo):
        bootstrap_masrouter_imports(repo)
        from MAR.Agent.reasoning_profile import reasoning_profile
        from MAR.LLM.llm_profile import llm_profile
        from MAR.MasRouter.mas_router import MasRouter
        from MAR.Prompts.tasks_profile import tasks_profile
        from Datasets.gsm8k_dataset import gsm_get_predict

        patch_masrouter_local_backend(llm_generate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        router = MasRouter(max_agent=3, device=device).to(device)
        optimizer = torch.optim.Adam(
            router.parameters(),
            lr=float(config.get("optimizer", {}).get("learning_rate", 0.01)),
        )
        epochs = int(config.get("optimizer", {}).get("local_iterations", 5))
        batch_size = max(1, min(len(examples), int(config.get("optimizer", {}).get("local_batch_size", 1))))
        cost_rate = 200.0 if client["dataset"] == "gsm_hard" else 0.05
        num_steps = 0
        for step, (epoch, batch) in enumerate(epoch_batches(examples, batch_size, epochs)):
            queries, answers, task_labels, prompt_file = masrouter_batch(client, batch)
            tasks_y = torch.tensor(task_labels, device=device)
            optimizer.zero_grad()
            results, costs, log_probs, tasks_probs, vae_loss, _ = router.forward(
                queries,
                tasks_profile,
                llm_profile,
                reasoning_profile,
                task_labels,
                prompt_file=prompt_file,
            )
            task_loss = F.cross_entropy(tasks_probs, tasks_y)
            answer_losses = []
            correct = []
            for result, answer, log_prob, cost in zip(results, answers, log_probs, costs):
                if client["dataset"] == "gsm_hard":
                    prediction = safe_gsm_prediction(str(result), gsm_get_predict)
                    ok = numeric_equal(prediction, answer)
                else:
                    prediction = normalize_prediction(client["dataset"], str(result))
                    ok = is_correct(client["dataset"], prediction, answer)
                utility = float(ok) - float(cost) * cost_rate
                answer_losses.append(-log_prob * utility)
                correct.append(float(ok))
            answer_loss = torch.stack(answer_losses).mean()
            loss = task_loss + answer_loss + vae_loss.mean() * 0.001
            loss.backward()
            optimizer.step()
            num_steps = step + 1

        checkpoint = {
            "format_version": 1,
            "kind": "masrouter_state_dict",
            "round": 0,
            "client_id": client["id"],
            "mas": client["mas"],
            "dataset": client["dataset"],
            "router_state_dict": router.state_dict(),
        }
    checkpoint_path = checkpoint_dir / "masrouter_round0.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, {
        "trainer": "MASRouter Experiments train loop",
        "num_examples": len(examples),
        "epochs": epochs,
        "batch_size": batch_size,
        "num_train_steps": num_steps,
    }


def train_client_native(
    args: argparse.Namespace,
    config: dict[str, Any],
    client: dict[str, Any],
    examples: list[dict[str, Any]],
    checkpoint_dir: Path,
    llm_generate,
) -> tuple[Path, dict[str, Any]]:
    if client["mas"] == "gptswarm":
        return train_gptswarm_native(args, config, client, examples, checkpoint_dir, llm_generate)
    if client["mas"] == "masrouter":
        return train_masrouter_native(args, config, client, examples, checkpoint_dir, llm_generate)
    raise ValueError(f"Unsupported native trainer for {client['mas']}")


def train_artifact_placeholder(
    config: dict[str, Any],
    client: dict[str, Any],
    num_examples: int,
    checkpoint_path: Path,
    rng: random.Random,
    iterations: int,
    *,
    round_index: int,
    prompt_state: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    state = build_initial_state(config, client, num_examples)
    state["round"] = round_index
    if prompt_state is not None:
        state.update(
            {
                "soft_prompt": prompt_state["soft_prompt"].detach().cpu(),
                "prompt_source": prompt_state.get("source_report"),
                "num_virtual_tokens": prompt_state.get("num_virtual_tokens"),
                "projection_seed": prompt_state.get("projection_seed"),
            }
        )
    best_objective = None
    for step in range(iterations):
        objective = max(0.0, 1.0 - 0.03 * step + rng.uniform(-0.01, 0.01))
        best_objective = objective if best_objective is None else min(best_objective, objective)
    state["metrics"] = {
        "best_objective": best_objective,
        "num_optimization_steps": iterations,
    }
    torch.save(state, checkpoint_path)
    return checkpoint_path, {
        "trainer": "artifact_placeholder",
        "best_objective": best_objective,
        "num_optimization_steps": iterations,
    }


def train_client_with_prompt(
    args: argparse.Namespace,
    config: dict[str, Any],
    client: dict[str, Any],
    train_split: str,
    train_examples: list[dict[str, Any]],
    client_dir: Path,
    backend: dict[str, Any],
    prompt_state: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    checkpoint_dir = client_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = hydra_payload(config, client, round_index, train_split, len(train_examples))
    save_hydra_files(client_dir, payload)

    llm_generate = lambda prompt: generate_with_embedding_soft_prompt(
        backend,
        prompt,
        args.max_new_tokens,
        prompt_state,
        config,
    )
    optimizer_cfg = config.get("optimizer", {})
    iterations = int(optimizer_cfg.get("local_iterations", 5))
    seed = int(config.get("seed", 42)) + round_index * 10007
    rng = random.Random(seed)
    if args.round0_trainer == "native":
        if not train_examples:
            raise RuntimeError(f"No training examples loaded for {client['id']}")
        checkpoint_path, train_metadata = train_client_native(
            args,
            config,
            client,
            train_examples,
            checkpoint_dir,
            llm_generate,
        )
    else:
        checkpoint_path = checkpoint_dir / f"model_round{round_index}.pt"
        checkpoint_path, train_metadata = train_artifact_placeholder(
            config,
            client,
            len(train_examples),
            checkpoint_path,
            rng,
            iterations,
            round_index=round_index,
            prompt_state=prompt_state,
        )
        train_metadata["trainer"] = "artifact_placeholder_with_embedding_soft_prompt"

    return {
        "client_id": client["id"],
        "mas": client["mas"],
        "dataset": client["dataset"],
        "dataset_name": client["dataset_name"],
        "split": train_split,
        "num_examples": len(train_examples),
        "hydra_run_dir": str(client_dir),
        "checkpoint_path": str(checkpoint_path),
        "trainer": train_metadata["trainer"],
        "train_metadata": train_metadata
        | {
            "round": round_index,
            "embedding_soft_prompt_enabled": True,
            "prompt_artifact_path": str(client_dir / f"prompt_round{round_index}.pt"),
            "num_virtual_tokens": prompt_state.get("num_virtual_tokens"),
            "projection_seed": prompt_state.get("projection_seed"),
        },
    }


def hydra_payload(
    config: dict[str, Any],
    client: dict[str, Any],
    round_index: int,
    split: str,
    num_examples: int,
) -> dict[str, Any]:
    return {
        "round": round_index,
        "stage": "round0_train" if round_index == 0 else f"round{round_index}",
        "seed": config.get("seed", 42),
        "client": client,
        "model": config.get("model", {}),
        "prompt": config.get("prompt", {}),
        "optimizer": config.get("optimizer", {}),
        "objective": config.get("objective", {}),
        "dataset": {
            "name": client["dataset_name"],
            "alias": client["dataset"],
            "split": split,
            "num_examples": num_examples,
        },
    }


def save_hydra_files(run_dir: Path, payload: dict[str, Any]) -> None:
    hydra_dir = run_dir / ".hydra"
    write_yaml(hydra_dir / "config.yaml", payload)
    write_yaml(
        hydra_dir / "hydra.yaml",
        {
            "run": {"dir": str(run_dir)},
            "job": {
                "name": "fedpao_round0_train",
                "chdir": False,
            },
        },
    )
    write_yaml(
        hydra_dir / "overrides.yaml",
        [
            f"round={payload['round']}",
            f"client.id={payload['client']['id']}",
            f"client.mas={payload['client']['mas']}",
            f"client.dataset={payload['client']['dataset']}",
        ],
    )


def build_initial_state(
    config: dict[str, Any],
    client: dict[str, Any],
    num_examples: int,
) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    client_seed = seed + sum(ord(char) for char in client["id"])
    generator = torch.Generator().manual_seed(client_seed)
    prompt_cfg = config.get("prompt", {})
    latent_dim = int(prompt_cfg.get("latent_dim", 32))
    init_std = float(prompt_cfg.get("init_std", 0.02))
    soft_prompt = torch.randn(latent_dim, generator=generator) * init_std
    return {
        "format_version": 1,
        "round": 0,
        "client_id": client["id"],
        "mas": client["mas"],
        "dataset": client["dataset"],
        "dataset_name": client["dataset_name"],
        "num_examples": num_examples,
        "prompt_type": prompt_cfg.get("type", "soft_latent"),
        "latent_dim": latent_dim,
        "soft_prompt": soft_prompt,
        "metrics": {},
    }


def validate_round0_client(
    args: argparse.Namespace,
    client: dict[str, Any],
    valid_split: str,
    valid_examples: list[dict[str, Any]],
    checkpoint_path: Path,
    backend: dict[str, Any],
) -> dict[str, Any]:
    if not valid_examples:
        return {"split": valid_split, "num_examples": 0, "accuracy": None}

    orchestrator = create_orchestrator(
        client["mas"],
        third_party_dir=args.third_party_dir,
        mode=args.orchestrator_mode,
        round0_checkpoint_path=str(checkpoint_path),
    )
    llm_generate = lambda prompt: generate(backend, prompt, args.max_new_tokens)
    correct = 0
    for example in valid_examples:
        task_prompt, ground_truth = make_prompt(client, example)
        orchestrated = orchestrator.run_inference(
            client,
            dict(example),
            task_prompt,
            llm_generate,
        )
        response = orchestrated.metadata.get("response", "")
        prediction = normalize_prediction(client["dataset"], response)
        ok = is_correct(client["dataset"], prediction, ground_truth)
        correct += int(ok)

    accuracy = correct / len(valid_examples)
    return {
        "split": valid_split,
        "num_examples": len(valid_examples),
        "accuracy": accuracy,
        "correct": correct,
    }


def round0_train(
    args: argparse.Namespace,
    config: dict[str, Any],
    clients: list[dict[str, Any]],
    run_root: Path,
    max_train_samples: int | None,
    backend: dict[str, Any] | None,
) -> dict[str, Any]:
    round_dir = run_root / "round_000"
    manifest: dict[str, Any] = {
        "round": 0,
        "stage": "train",
        "created_at": timestamp(),
        "clients": [],
    }
    optimizer_cfg = config.get("optimizer", {})
    iterations = int(optimizer_cfg.get("local_iterations", 5))
    rng = random.Random(int(config.get("seed", 42)))
    data_cfg = config.get("data", {})
    train_fraction = float(data_cfg.get("train_fraction", 0.8))
    sample_limit = data_cfg.get("max_samples_per_client")
    sample_limit = int(sample_limit) if sample_limit is not None else None
    valid_limit = args.max_valid_samples_per_client

    for client in clients:
        split, examples, valid_split, valid_examples = split_client_examples(
            client,
            train_fraction,
            int(config.get("seed", 42)),
            sample_limit=sample_limit,
            train_limit=max_train_samples,
            valid_limit=valid_limit,
        )
        num_examples = len(examples)
        client_name = slugify(f"{client['id']}_{client['mas']}_{client['dataset']}")
        client_dir = round_dir / client_name
        checkpoint_dir = client_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if args.round0_trainer == "native":
            if backend is None:
                raise RuntimeError("native round0 training requires an LLM backend")
            if not examples:
                raise RuntimeError(f"No training examples loaded for {client['id']}")
            payload = hydra_payload(config, client, 0, split, num_examples)
            save_hydra_files(client_dir, payload)
            llm_generate = lambda prompt: generate(backend, prompt, args.max_new_tokens)
            checkpoint_path, train_metadata = train_client_native(
                args,
                config,
                client,
                examples,
                checkpoint_dir,
                llm_generate,
            )
            if args.max_valid_samples_per_client != 0:
                train_metadata["validation"] = validate_round0_client(
                    args,
                    client,
                    valid_split,
                    valid_examples,
                    checkpoint_path,
                    backend,
                )
        else:
            payload = hydra_payload(config, client, 0, split, num_examples)
            save_hydra_files(client_dir, payload)
            checkpoint_path = checkpoint_dir / "model_round0.pt"
            checkpoint_path, train_metadata = train_artifact_placeholder(
                config,
                client,
                num_examples,
                checkpoint_path,
                rng,
                iterations,
                round_index=0,
            )
            if args.max_valid_samples_per_client != 0 and backend is not None:
                train_metadata["validation"] = validate_round0_client(
                    args,
                    client,
                    valid_split,
                    valid_examples,
                    checkpoint_path,
                    backend,
                )

        client_record = {
            "client_id": client["id"],
            "mas": client["mas"],
            "dataset": client["dataset"],
            "dataset_name": client["dataset_name"],
            "split": split,
            "valid_split": valid_split,
            "num_examples": num_examples,
            "num_valid_examples": len(valid_examples),
            "hydra_run_dir": str(client_dir),
            "checkpoint_path": str(checkpoint_path),
            "trainer": train_metadata["trainer"],
            "train_metadata": train_metadata,
        }
        write_json(client_dir / "manifest.json", client_record)
        manifest["clients"].append(client_record)
        print(
            f"round0 trained {client['id']} {client['mas']} {client['dataset']} "
            f"-> {checkpoint_path}"
        )

    write_json(round_dir / "manifest.json", manifest)
    return manifest


def load_round0_models(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    loaded = {}
    for record in manifest["clients"]:
        checkpoint_path = Path(record["checkpoint_path"])
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
        loaded[record["client_id"]] = {
            "record": record,
            "state": state,
        }
    return loaded


def compact_text(value: str, limit: int = 4000) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def filter_third_party_execution_noise(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    filtered = []
    suppressed = 0
    for line in lines:
        if line.startswith("Error during execution of node "):
            suppressed += 1
            continue
        filtered.append(line)
    return "".join(filtered), suppressed


def question_type(client: dict[str, Any], example: dict[str, Any]) -> str:
    if client["dataset"] == "mmlu_pro":
        return str(example.get("category") or example.get("subject") or "multiple_choice")
    if client["dataset"] == "gsm_hard":
        text = str(example.get("input", ""))
        if any(token in text.lower() for token in ("percent", "%", "ratio", "fraction")):
            return "quantitative_ratio_or_percent"
        if any(token in text.lower() for token in ("how many", "total", "sum", "altogether")):
            return "quantitative_counting"
        return "quantitative_reasoning"
    return client["dataset"]


def trajectory_from_run(
    *,
    client: dict[str, Any],
    example: dict[str, Any],
    task_prompt: str,
    ground_truth: str,
    orchestrated,
    response: str,
    prediction: str,
    correct: bool,
    latency: float,
    stdout_text: str,
    stderr_text: str,
) -> dict[str, Any]:
    metadata = orchestrated.metadata
    return {
        "input": task_prompt,
        "question_type": question_type(client, example),
        "orchestrator": client["mas"],
        "native_entrypoint": metadata.get("native_entrypoint"),
        "agent_call_summary": {
            "agents_num": metadata.get("agents_num"),
            "cost": metadata.get("cost"),
            "swarm_agents": metadata.get("swarm_agents"),
        },
        "orchestrator_metadata": metadata,
        "terminal_stdout": compact_text(stdout_text),
        "terminal_stderr": compact_text(stderr_text),
        "response": response,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "correct": correct,
        "latency_sec": round(latency, 4),
    }


def build_summary_agent_prompt(client: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    observations = []
    for row in rows:
        trajectory = row["trajectory"]
        observations.append(
            {
                "question_type": trajectory["question_type"],
                "orchestrator": trajectory["orchestrator"],
                "agent_call_summary": trajectory["agent_call_summary"],
                "prediction": row["prediction"],
                "ground_truth": row["ground_truth"],
                "correct": row["correct"],
                "latency_sec": row["latency_sec"],
                "terminal_stdout": trajectory["terminal_stdout"],
                "terminal_stderr": trajectory["terminal_stderr"],
            }
        )
    return (
        """
            You are a routing analyst.

            Your goal is NOT to summarize accuracy, latency, token cost, or agent usage statistics.

            Instead, infer routing patterns from the trajectory that could help a future orchestrator decide which agents to call.

            Focus on:

            1. What characteristics of the user query triggered successful agent selections.
            2. What agent sequences or combinations appeared useful.
            3. What agent calls were unnecessary because they did not contribute new information.
            4. One routing rule that a future orchestrator could directly apply.

            Describe behavioral patterns rather than numerical metrics.

            Write at most 3 sentences in natural language. Do not use bullets.\n\n
        """
        f"Client: {client['id']} / {client['mas']} / {client['dataset']}\n"
        f"Trajectory observations:\n{json.dumps(observations, ensure_ascii=False, indent=2)}"
    )


def trim_to_sentences(text: str, max_sentences: int = 3) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?。！？다])\s+", text)
    selected = [part for part in parts if part][:max_sentences]
    return " ".join(selected).strip()


def first_backend_device(backend: dict[str, Any]) -> torch.device:
    try:
        return next(backend["model"].parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def backend_tokenizer(backend: dict[str, Any]):
    if backend["kind"] == "causal_lm":
        return backend["tokenizer"]
    processor = backend.get("processor")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "Cannot build report sentence embeddings: backend processor has no tokenizer"
        )
    return tokenizer


def sentence_embedding_from_report(
    backend: dict[str, Any],
    text: str,
    max_length: int = 512,
) -> torch.Tensor:
    """Mean-pool frozen input-token embeddings for a report sentence embedding."""

    tokenizer = backend_tokenizer(backend)
    model = backend["model"]
    encoded = tokenizer(
        text or "empty routing report",
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    device = first_backend_device(backend)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)

    embedding_layer = model.get_input_embeddings()
    with torch.inference_mode():
        token_embeddings = embedding_layer(input_ids).float()
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled.squeeze(0).detach().cpu()


def project_report_embedding_to_latent(
    embedding: torch.Tensor,
    latent_dim: int,
    seed: int,
    init_std: float,
) -> torch.Tensor:
    embedding = embedding.float().flatten()
    if embedding.numel() == latent_dim:
        latent = embedding
    else:
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(
            embedding.numel(),
            latent_dim,
            generator=generator,
            dtype=torch.float32,
        ) / max(embedding.numel(), 1) ** 0.5
        latent = embedding @ projection
    latent = F.normalize(latent, dim=0) * (latent_dim ** 0.5) * init_std
    return latent.detach().cpu()


def score_prompt_candidate(
    candidate: torch.Tensor,
    report_latent: torch.Tensor,
    report: dict[str, Any],
    objective_cfg: dict[str, Any],
) -> float:
    alignment = F.cosine_similarity(candidate, report_latent, dim=0).item()
    accuracy = float(report.get("accuracy") or 0.0)
    latent_norm_weight = float(objective_cfg.get("latent_norm_weight", 0.0))
    norm_penalty = candidate.norm().item()
    return (
        alignment
        + accuracy
        - latent_norm_weight * norm_penalty
    )


def gradient_free_refine_prompt_latent(
    base_latent: torch.Tensor,
    report: dict[str, Any],
    config: dict[str, Any],
    client: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    optimizer_cfg = config.get("optimizer", {})
    prompt_cfg = config.get("prompt", {})
    iterations = int(optimizer_cfg.get("local_iterations", 5))
    population_size = int(optimizer_cfg.get("population_size", 8))
    sigma0 = float(optimizer_cfg.get("sigma0", 0.5))
    init_std = float(prompt_cfg.get("init_std", 0.02))
    seed = int(config.get("seed", 42)) + sum(ord(char) for char in client["id"]) + 1009
    generator = torch.Generator().manual_seed(seed)
    objective_cfg = config.get("objective", {})
    optimizer_name = str(optimizer_cfg.get("name", "gradient_free_random_search")).lower()

    if optimizer_name in {"cma_es", "cma-es", "cma"}:
        try:
            import cma
        except ImportError as exc:
            raise RuntimeError(
                "optimizer.name is cma_es, but the `cma` package is not installed. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        base_np = base_latent.detach().float().cpu().numpy()
        es = cma.CMAEvolutionStrategy(
            base_np,
            sigma0 * init_std,
            {
                "popsize": population_size,
                "seed": seed,
                "maxiter": iterations,
                "verbose": -9,
            },
        )
        best_latent = base_latent.clone().float()
        best_score = score_prompt_candidate(
            best_latent,
            base_latent,
            report,
            objective_cfg,
        )
        history = [{"iteration": 0, "best_score": best_score}]

        for iteration in range(1, iterations + 1):
            candidates = es.ask()
            losses = []
            for candidate_np in candidates:
                candidate = torch.from_numpy(
                    np.asarray(candidate_np, dtype=np.float32)
                )
                score = score_prompt_candidate(
                    candidate,
                    base_latent,
                    report,
                    objective_cfg,
                )
                losses.append(-score)
                if score > best_score:
                    best_score = score
                    best_latent = candidate.detach().cpu()
            es.tell(candidates, losses)
            history.append({"iteration": iteration, "best_score": best_score})

        return best_latent, {
            "optimizer": "cma_es",
            "iterations": iterations,
            "population_size": population_size,
            "sigma0": sigma0,
            "seed": seed,
            "objective": "maximize cosine(report_latent, candidate) + accuracy - latent norm penalty",
            "best_score": best_score,
            "history": history,
        }

    best_latent = base_latent.clone().float()
    best_score = score_prompt_candidate(best_latent, base_latent, report, objective_cfg)
    history = [{"iteration": 0, "best_score": best_score}]
    noise_scale = sigma0 * init_std

    for iteration in range(1, iterations + 1):
        for _ in range(population_size):
            candidate = base_latent + torch.randn(
                base_latent.shape,
                generator=generator,
                dtype=torch.float32,
            ) * noise_scale
            score = score_prompt_candidate(candidate, base_latent, report, objective_cfg)
            if score > best_score:
                best_score = score
                best_latent = candidate.detach().cpu()
        history.append({"iteration": iteration, "best_score": best_score})

    return best_latent, {
        "optimizer": optimizer_name,
        "iterations": iterations,
        "population_size": population_size,
        "sigma0": sigma0,
        "seed": seed,
        "objective": "cosine(report_latent, candidate) + accuracy - latent norm penalty",
        "best_score": best_score,
        "history": history,
    }


def prompt_bridge_text(report: dict[str, Any]) -> str:
    summary = report.get("summary", "").strip()
    if not summary:
        return ""
    return (
        "Previous-round routing guidance from the summary agent:\n"
        f"{summary}\n"
        "Use this guidance when choosing agent calls, but still return only the task answer."
    )


def load_global_prompt_state(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Global prompt artifact does not exist: {artifact_path}")
    try:
        state = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(artifact_path, map_location="cpu")
    if "soft_prompt" not in state:
        raise ValueError(f"Global prompt artifact has no soft_prompt tensor: {artifact_path}")
    return state


def latent_to_virtual_prompt_embeddings(
    soft_prompt: torch.Tensor,
    *,
    hidden_size: int,
    num_virtual_tokens: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    soft_prompt = soft_prompt.detach().float().cpu()
    if soft_prompt.ndim == 2:
        if soft_prompt.shape != (num_virtual_tokens, hidden_size):
            raise ValueError(
                "2D soft_prompt must have shape "
                f"({num_virtual_tokens}, {hidden_size}), got {tuple(soft_prompt.shape)}"
            )
        return soft_prompt.to(device=device, dtype=dtype)

    latent = soft_prompt.flatten()
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(
        latent.numel(),
        num_virtual_tokens * hidden_size,
        generator=generator,
        dtype=torch.float32,
    ) / max(latent.numel(), 1) ** 0.5
    virtual_prompt = latent @ projection
    virtual_prompt = virtual_prompt.reshape(num_virtual_tokens, hidden_size)
    return virtual_prompt.to(device=device, dtype=dtype)


def generate_with_embedding_soft_prompt(
    backend: dict[str, Any],
    prompt: str,
    max_new_tokens: int,
    global_prompt_state: dict[str, Any],
    config: dict[str, Any],
) -> str:
    if backend["kind"] != "causal_lm":
        raise ValueError(
            "Embedding-level soft prompt injection currently requires a causal_lm "
            "backend because image_text processors own multimodal embedding assembly."
        )

    model = backend["model"]
    tokenizer = backend["tokenizer"]
    device = first_backend_device(backend)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    embedding_layer = model.get_input_embeddings()
    token_embeddings = embedding_layer(input_ids)
    prompt_cfg = config.get("prompt", {})
    virtual_tokens_value = global_prompt_state.get("num_virtual_tokens")
    if virtual_tokens_value is None:
        virtual_tokens_value = prompt_cfg.get("num_virtual_tokens", prompt_cfg.get("soft_prompt_tokens", 8))
    num_virtual_tokens = int(virtual_tokens_value)
    projection_seed_value = global_prompt_state.get("projection_seed")
    if projection_seed_value is None:
        projection_seed_value = int(config.get("seed", 42)) + 917
    projection_seed = int(projection_seed_value)
    soft_embeddings = latent_to_virtual_prompt_embeddings(
        global_prompt_state["soft_prompt"],
        hidden_size=token_embeddings.shape[-1],
        num_virtual_tokens=num_virtual_tokens,
        seed=projection_seed,
        dtype=token_embeddings.dtype,
        device=device,
    ).unsqueeze(0)

    inputs_embeds = torch.cat([soft_embeddings, token_embeddings], dim=1)
    soft_attention = torch.ones(
        input_ids.shape[0],
        num_virtual_tokens,
        dtype=attention_mask.dtype,
        device=device,
    )
    combined_attention = torch.cat([soft_attention, attention_mask], dim=1)
    prefix_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else 0
    )
    prefix_ids = torch.full(
        (input_ids.shape[0], num_virtual_tokens),
        int(prefix_token_id),
        dtype=input_ids.dtype,
        device=device,
    )
    combined_input_ids = torch.cat([prefix_ids, input_ids], dim=1)

    with torch.inference_mode():
        try:
            output = model.generate(
                input_ids=combined_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=combined_attention,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = output[0][combined_input_ids.shape[-1]:]
        except (TypeError, ValueError):
            output = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_attention,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = output[0]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def write_client_prompt_artifact(
    client_dir: Path,
    client: dict[str, Any],
    report: dict[str, Any],
    backend: dict[str, Any],
    config: dict[str, Any],
    round_index: int = 1,
) -> dict[str, Any]:
    prompt_cfg = config.get("prompt", {})
    latent_dim = int(prompt_cfg.get("latent_dim", 32))
    init_std = float(prompt_cfg.get("init_std", 0.02))
    num_virtual_tokens = int(prompt_cfg.get("num_virtual_tokens", prompt_cfg.get("soft_prompt_tokens", 8)))
    projection_seed = int(config.get("seed", 42)) + 313
    report_embedding = sentence_embedding_from_report(backend, report["summary"])
    report_latent = project_report_embedding_to_latent(
        report_embedding,
        latent_dim,
        projection_seed,
        init_std,
    )
    soft_prompt, optimization = gradient_free_refine_prompt_latent(
        report_latent,
        report,
        config,
        client,
    )

    artifact_path = client_dir / f"prompt_round{round_index}.pt"
    manifest_path = client_dir / "prompt_manifest.json"
    torch.save(
        {
            "format_version": 1,
            "round": round_index,
            "client_id": client["id"],
            "mas": client["mas"],
            "dataset": client["dataset"],
            "prompt_type": prompt_cfg.get("type", "soft_latent"),
            "latent_dim": latent_dim,
            "num_virtual_tokens": num_virtual_tokens,
            "projection_seed": projection_seed,
            "report_embedding": report_embedding,
            "report_latent": report_latent,
            "soft_prompt": soft_prompt,
            "bridge_text": prompt_bridge_text(report),
            "source_report": str(client_dir / "analysis_report.json"),
            "optimization": optimization,
            "metrics": {
                "num_examples": report["num_examples"],
                "correct": report["correct"],
                "accuracy": report["accuracy"],
            },
        },
        artifact_path,
    )
    manifest = {
        "round": round_index,
        "client_id": client["id"],
        "mas": client["mas"],
        "dataset": client["dataset"],
        "prompt_type": prompt_cfg.get("type", "soft_latent"),
        "latent_dim": latent_dim,
        "num_virtual_tokens": num_virtual_tokens,
        "projection_seed": projection_seed,
        "num_examples": report["num_examples"],
        "accuracy": report["accuracy"],
        "prompt_artifact_path": str(artifact_path),
        "source_report": str(client_dir / "analysis_report.json"),
        "bridge_text": prompt_bridge_text(report),
        "optimization": optimization,
    }
    write_json(manifest_path, manifest)
    return manifest


def fedavg_prompt_artifacts(
    round_dir: Path,
    prompt_manifests: list[dict[str, Any]],
    config: dict[str, Any],
    source_round: int = 1,
) -> dict[str, Any] | None:
    if not prompt_manifests:
        return None

    federated_cfg = config.get("federated", {})
    weight_by = federated_cfg.get("weight_by", "num_examples")
    weighted_latents = []
    clients = []
    bridge_texts = []
    total_weight = 0.0
    latent_dim = None

    for manifest in prompt_manifests:
        try:
            state = torch.load(
                manifest["prompt_artifact_path"],
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            state = torch.load(manifest["prompt_artifact_path"], map_location="cpu")
        soft_prompt = state["soft_prompt"].float()
        if latent_dim is None:
            latent_dim = int(soft_prompt.numel())
        if int(soft_prompt.numel()) != latent_dim:
            raise ValueError("Cannot FedAvg prompt artifacts with different latent dimensions")
        weight = float(manifest.get(weight_by) or 1.0)
        weighted_latents.append(soft_prompt * weight)
        total_weight += weight
        clients.append(
            {
                "client_id": manifest["client_id"],
                "weight": weight,
                "prompt_artifact_path": manifest["prompt_artifact_path"],
                "accuracy": manifest["accuracy"],
            }
        )
        if manifest.get("bridge_text"):
            bridge_texts.append(
                {
                    "client_id": manifest["client_id"],
                    "bridge_text": manifest["bridge_text"],
                }
            )

    if total_weight <= 0.0:
        total_weight = float(len(weighted_latents))
        weighted_latents = []
        for manifest in prompt_manifests:
            try:
                state = torch.load(
                    manifest["prompt_artifact_path"],
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                state = torch.load(manifest["prompt_artifact_path"], map_location="cpu")
            weighted_latents.append(state["soft_prompt"].float())
    global_soft_prompt = torch.stack(weighted_latents).sum(dim=0) / total_weight
    server_dir = round_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    target_round = source_round + 1
    artifact_path = server_dir / f"global_prompt_round{target_round}.pt"
    manifest_path = server_dir / "global_prompt_manifest.json"
    torch.save(
        {
            "format_version": 1,
            "source_round": source_round,
            "target_round": target_round,
            "aggregation": federated_cfg.get("aggregation", "fedavg"),
            "weight_by": weight_by,
            "latent_dim": latent_dim,
            "num_virtual_tokens": prompt_manifests[0].get("num_virtual_tokens", 8),
            "projection_seed": prompt_manifests[0].get("projection_seed"),
            "soft_prompt": global_soft_prompt.detach().cpu(),
            "bridge_texts": bridge_texts,
            "clients": clients,
        },
        artifact_path,
    )
    manifest = {
        "source_round": source_round,
        "target_round": target_round,
        "aggregation": federated_cfg.get("aggregation", "fedavg"),
        "weight_by": weight_by,
        "num_clients": len(prompt_manifests),
        "total_weight": total_weight,
        "latent_dim": latent_dim,
        "num_virtual_tokens": prompt_manifests[0].get("num_virtual_tokens", 8),
        "projection_seed": prompt_manifests[0].get("projection_seed"),
        "global_prompt_artifact_path": str(artifact_path),
        "bridge_texts": bridge_texts,
        "clients": clients,
    }
    write_json(manifest_path, manifest)
    return manifest


def write_client_analysis_report(
    client_dir: Path,
    client: dict[str, Any],
    rows: list[dict[str, Any]],
    llm_generate,
) -> dict[str, Any]:
    prompt = build_summary_agent_prompt(client, rows)
    summary = trim_to_sentences(llm_generate(prompt), 3)
    correct = sum(row["correct"] for row in rows)
    report = {
        "client_id": client["id"],
        "mas": client["mas"],
        "dataset": client["dataset"],
        "num_examples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "summary_agent_prompt": prompt,
        "summary": summary,
        "examples": rows,
    }
    write_json(client_dir / "analysis_report.json", report)
    with (client_dir / "analysis_report.md").open("w", encoding="utf-8") as file:
        file.write(f"# Round 1 Analysis: {client['id']}\n\n")
        file.write(summary + "\n\n")
        file.write(f"- MAS: {client['mas']}\n")
        file.write(f"- Dataset: {client['dataset']}\n")
        file.write(f"- Accuracy: {correct}/{len(rows)}\n")
    return report


def resolve_runtime_args(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.model_path is None:
        args.model_path = config.get("model", {}).get("model_id", DEFAULT_MODEL_PATH)


def resolve_run_root(args: argparse.Namespace) -> tuple[str, Path]:
    if args.run_root:
        run_root = Path(args.run_root).resolve()
        run_name = args.run_name or run_root.name
        return run_name, run_root
    if args.start_round > 0 and not args.run_name:
        raise ValueError("--start-round 1 requires --run-name or --run-root so round0 artifacts can be reused")
    run_name = args.run_name or f"fedpao_{timestamp()}"
    return run_name, (Path(args.output_dir) / slugify(run_name)).resolve()


def load_round0_manifest_for_run(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "round_000" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Could not find round0 manifest for resume: {manifest_path}"
        )
    manifest = read_json(manifest_path)
    return rebase_round0_manifest_paths(manifest, run_root)


def load_round_manifest_for_run(run_root: Path, round_index: int) -> dict[str, Any]:
    manifest_path = run_root / f"round_{round_index:03d}" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Could not find round {round_index} manifest for resume: {manifest_path}"
        )
    manifest = read_json(manifest_path)
    return rebase_round0_manifest_paths(manifest, run_root)


def rebase_path_to_run_root(path_value: str, run_root: Path) -> str:
    path = Path(path_value)
    if path.exists():
        return path_value
    parts = path.parts
    for idx in range(len(parts) - 2):
        if parts[idx : idx + 3] == ("outputs", "runs", run_root.name):
            relative = Path(*parts[idx + 3 :])
            candidate = run_root / relative
            return str(candidate)
    return path_value


def rebase_round0_manifest_paths(
    manifest: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    path_keys = (
        "hydra_run_dir",
        "checkpoint_path",
        "previous_checkpoint_path",
        "analysis_report_path",
        "prompt_artifact_path",
    )
    for record in manifest.get("clients", []):
        for key in path_keys:
            if key in record and isinstance(record[key], str):
                record[key] = rebase_path_to_run_root(record[key], run_root)
    global_prompt = manifest.get("global_prompt")
    if isinstance(global_prompt, dict):
        key = "global_prompt_artifact_path"
        if isinstance(global_prompt.get(key), str):
            global_prompt[key] = rebase_path_to_run_root(global_prompt[key], run_root)
    return manifest


def run_orchestrator_examples(
    *,
    args: argparse.Namespace,
    client: dict[str, Any],
    examples: list[dict[str, Any]],
    split: str,
    role: str,
    round_index: int,
    orchestrator,
    llm_generate,
    model_path: str,
    record: dict[str, Any],
    state: dict[str, Any],
    global_prompt_artifact: str | None,
    global_prompt_enabled: bool,
    output_file,
) -> list[dict[str, Any]]:
    rows = []
    for idx, example in enumerate(examples):
        task_prompt, ground_truth = make_prompt(client, example)
        start = time.perf_counter()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout_buffer),
            contextlib.redirect_stderr(stderr_buffer),
        ):
            orchestrated = orchestrator.run_inference(
                client,
                dict(example),
                task_prompt,
                llm_generate,
            )
        latency = time.perf_counter() - start
        stdout_text = stdout_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()
        printable_stdout, suppressed_stdout_errors = filter_third_party_execution_noise(
            stdout_text
        )
        printable_stderr, suppressed_stderr_errors = filter_third_party_execution_noise(
            stderr_text
        )
        suppressed_execution_errors = suppressed_stdout_errors + suppressed_stderr_errors
        if printable_stdout:
            print(printable_stdout, end="")
        if printable_stderr:
            print(printable_stderr, end="", file=sys.stderr)
        if suppressed_execution_errors:
            print(
                f"round{round_index} {client['id']} suppressed "
                f"{suppressed_execution_errors} third-party node retry errors; "
                "full text is saved in trajectories.jsonl"
            )
        response = orchestrated.metadata.get("response", "")
        prediction = normalize_prediction(client["dataset"], response)
        correct = is_correct(client["dataset"], prediction, ground_truth)
        trajectory = trajectory_from_run(
            client=client,
            example=dict(example),
            task_prompt=task_prompt,
            ground_truth=ground_truth,
            orchestrated=orchestrated,
            response=response,
            prediction=prediction,
            correct=correct,
            latency=latency,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        row = {
            "round": round_index,
            "client_id": client["id"],
            "mas": client["mas"],
            "dataset": client["dataset"],
            "dataset_name": client["dataset_name"],
            "split": split,
            "role": role,
            "example_index": idx,
            "model_path": model_path,
            "previous_checkpoint_path": record["checkpoint_path"],
            "previous_hydra_run_dir": record["hydra_run_dir"],
            "previous_loaded_prompt_type": state.get("prompt_type"),
            "previous_loaded_latent_dim": state.get("latent_dim"),
            "orchestrator_mode": args.orchestrator_mode,
            "global_prompt_artifact": global_prompt_artifact,
            "embedding_soft_prompt_enabled": global_prompt_enabled,
            "suppressed_execution_errors": suppressed_execution_errors,
            "orchestrator_metadata": orchestrated.metadata,
            "trajectory": trajectory,
            "response": response,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": correct,
            "latency_sec": round(latency, 4),
        }
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_file.flush()
        rows.append(row)
        print(
            f"round{round_index} {role} {client['id']} {client['mas']} {client['dataset']}: "
            f"pred={prediction!r} gt={ground_truth!r} "
            f"correct={row['correct']} latency={row['latency_sec']}s"
        )
    return rows


def run_federated_round(
    args: argparse.Namespace,
    config: dict[str, Any],
    clients: list[dict[str, Any]],
    run_root: Path,
    previous_manifest: dict[str, Any],
    round_index: int,
    previous_global_prompt_path: str | None = None,
) -> dict[str, Any]:
    loaded_models = load_round0_models(previous_manifest)
    round_dir = run_root / f"round_{round_index:03d}"
    output_path = round_dir / "inference.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backend = load_backend(args.model_path, args.dtype)
    global_prompt_artifact = previous_global_prompt_path or args.global_prompt_artifact
    global_prompt_state = load_global_prompt_state(global_prompt_artifact)
    if global_prompt_state is None:
        llm_generate = lambda prompt: generate(backend, prompt, args.max_new_tokens)
    else:
        llm_generate = lambda prompt: generate_with_embedding_soft_prompt(
            backend,
            prompt,
            args.max_new_tokens,
            global_prompt_state,
            config,
        )
    summary_generate = lambda prompt: generate(backend, prompt, args.summary_max_new_tokens)
    rows = []
    reports = []
    prompt_manifests = []
    trained_clients = []

    with output_path.open("w", encoding="utf-8") as out:
        for client in clients:
            previous_model = loaded_models[client["id"]]
            record = previous_model["record"]
            state = previous_model["state"]
            orchestrator = create_orchestrator(
                client["mas"],
                third_party_dir=args.third_party_dir,
                mode=args.orchestrator_mode,
                round0_checkpoint_path=record["checkpoint_path"],
            )
            data_cfg = config.get("data", {})
            train_fraction = float(data_cfg.get("train_fraction", 0.8))
            sample_limit = data_cfg.get("max_samples_per_client")
            sample_limit = int(sample_limit) if sample_limit is not None else None
            trajectory_limit = (
                args.max_samples_per_client
                if args.max_samples_per_client is not None
                else data_cfg.get("max_trajectory_samples_per_client", 3)
            )
            trajectory_limit = (
                int(trajectory_limit) if trajectory_limit is not None else None
            )
            (
                train_split,
                train_examples,
                valid_split,
                valid_examples,
                trajectory_split,
                trajectory_examples,
            ) = split_client_examples_with_trajectory(
                client,
                train_fraction,
                int(config.get("seed", 42)),
                sample_limit=sample_limit,
                train_limit=args.max_train_samples_per_client,
                valid_limit=args.max_valid_samples_per_client,
                trajectory_limit=trajectory_limit,
            )
            if not valid_examples:
                raise RuntimeError(
                    "No validation examples available for round accuracy for "
                    f"{client['id']}. Set data.max_samples_per_client below the "
                    "dataset size or reduce split limits."
                )
            if not trajectory_examples:
                raise RuntimeError(
                    "No held-out examples available for trajectory analysis for "
                    f"{client['id']}. Set data.max_samples_per_client below the "
                    "dataset size or reduce split limits."
                )
            client_name = slugify(f"{client['id']}_{client['mas']}_{client['dataset']}")
            client_dir = round_dir / client_name
            client_dir.mkdir(parents=True, exist_ok=True)
            client_valid_rows = run_orchestrator_examples(
                args=args,
                client=client,
                examples=valid_examples,
                split=valid_split,
                role="validation",
                round_index=round_index,
                orchestrator=orchestrator,
                llm_generate=llm_generate,
                model_path=args.model_path,
                record=record,
                state=state,
                global_prompt_artifact=global_prompt_artifact,
                global_prompt_enabled=global_prompt_state is not None,
                output_file=out,
            )
            client_trajectory_rows = run_orchestrator_examples(
                args=args,
                client=client,
                examples=trajectory_examples,
                split=trajectory_split,
                role="summary_trajectory",
                round_index=round_index,
                orchestrator=orchestrator,
                llm_generate=llm_generate,
                model_path=args.model_path,
                record=record,
                state=state,
                global_prompt_artifact=global_prompt_artifact,
                global_prompt_enabled=global_prompt_state is not None,
                output_file=out,
            )
            rows.extend(client_valid_rows)
            with (client_dir / "trajectories.jsonl").open("w", encoding="utf-8") as file:
                for row in client_trajectory_rows:
                    file.write(json.dumps(row["trajectory"], ensure_ascii=False) + "\n")
            report = write_client_analysis_report(
                client_dir,
                client,
                client_trajectory_rows,
                summary_generate,
            )
            prompt_manifest = write_client_prompt_artifact(
                client_dir,
                client,
                report,
                backend,
                config,
                round_index=round_index,
            )
            prompt_state = load_global_prompt_state(prompt_manifest["prompt_artifact_path"])
            assert prompt_state is not None
            trained_record = train_client_with_prompt(
                args,
                config,
                client,
                train_split,
                train_examples,
                client_dir,
                backend,
                prompt_state,
                round_index,
            )
            trained_record.update(
                {
                    "valid_split": valid_split,
                    "num_valid_examples": len(valid_examples),
                    "trajectory_split": trajectory_split,
                    "num_trajectory_examples": len(trajectory_examples),
                    "valid_correct": sum(row["correct"] for row in client_valid_rows),
                    "valid_accuracy": (
                        sum(row["correct"] for row in client_valid_rows)
                        / len(client_valid_rows)
                        if client_valid_rows
                        else None
                    ),
                    "validation": {
                        "split": valid_split,
                        "num_examples": len(client_valid_rows),
                        "correct": sum(row["correct"] for row in client_valid_rows),
                        "accuracy": (
                            sum(row["correct"] for row in client_valid_rows)
                            / len(client_valid_rows)
                            if client_valid_rows
                            else None
                        ),
                    },
                    "summary_trajectory": {
                        "split": trajectory_split,
                        "num_examples": report["num_examples"],
                        "correct": report["correct"],
                        "accuracy": report["accuracy"],
                    },
                    "previous_checkpoint_path": record["checkpoint_path"],
                    "analysis_report_path": str(client_dir / "analysis_report.json"),
                    "prompt_artifact_path": prompt_manifest["prompt_artifact_path"],
                }
            )
            write_json(client_dir / "manifest.json", trained_record)
            trained_clients.append(trained_record)
            prompt_manifests.append(prompt_manifest)
            reports.append(
                {
                    "client_id": client["id"],
                    "report_path": str(client_dir / "analysis_report.json"),
                    "summary": report["summary"],
                    "valid_num_examples": len(client_valid_rows),
                    "valid_correct": sum(row["correct"] for row in client_valid_rows),
                    "valid_accuracy": (
                        sum(row["correct"] for row in client_valid_rows)
                        / len(client_valid_rows)
                        if client_valid_rows
                        else None
                    ),
                    "summary_trajectory_num_examples": report["num_examples"],
                    "summary_trajectory_accuracy": report["accuracy"],
                    "prompt_artifact_path": prompt_manifest["prompt_artifact_path"],
                    "trained_checkpoint_path": trained_record["checkpoint_path"],
                }
            )
            print(f"round{round_index} analysis {client['id']}: {report['summary']}")

    correct = sum(row["correct"] for row in rows)
    global_prompt_manifest = fedavg_prompt_artifacts(
        round_dir,
        prompt_manifests,
        config,
        source_round=round_index,
    )
    write_json(
        round_dir / "analysis_manifest.json",
        {
            "round": round_index,
            "num_clients": len(reports),
            "reports": reports,
            "global_prompt": global_prompt_manifest,
        },
    )
    round_manifest = {
        "round": round_index,
        "stage": "infer_analyze_prompt_train_aggregate",
        "created_at": timestamp(),
        "previous_global_prompt_artifact": global_prompt_artifact,
        "inference_path": str(output_path),
        "analysis_manifest_path": str(round_dir / "analysis_manifest.json"),
        "global_prompt": global_prompt_manifest,
        "clients": trained_clients,
        "accuracy": {
            "correct": correct,
            "num_examples": len(rows),
            "value": correct / len(rows) if rows else None,
        },
    }
    write_json(round_dir / "manifest.json", round_manifest)
    print(f"round{round_index} wrote: {output_path}")
    print(f"round{round_index} manifest: {round_dir / 'manifest.json'}")
    if global_prompt_manifest is not None:
        print(
            f"round{round_index} global prompt artifact: "
            f"{global_prompt_manifest['global_prompt_artifact_path']}"
        )
    print(f"round{round_index} accuracy: {correct}/{len(rows)}")
    return round_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FedPAO round runner: round 0 trains initial MAS artifacts; each "
            "later round infers, analyzes, builds soft prompts, retrains MAS, "
            "and FedAvg-aggregates the prompts."
        )
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--run-root",
        default=None,
        help="Existing or new run directory. Useful with --start-round 1 to reuse round0 artifacts.",
    )
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--model-path")
    parser.add_argument("--third-party-dir", default="third_party")
    parser.add_argument("--mas", choices=SUPPORTED_MAS, nargs="+")
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of post-training inference rounds to run. Use 1 for round 1.",
    )
    parser.add_argument(
        "--start-round",
        type=int,
        default=0,
        help=(
            "First round to execute. Use 1 to skip round0 and reuse round_000; "
            "use N>1 to resume from round_(N-1)."
        ),
    )
    parser.add_argument(
        "--orchestrator-mode",
        choices=["native", "simulated"],
        default="native",
    )
    parser.add_argument(
        "--round0-trainer",
        choices=["native", "artifact"],
        default="native",
        help="native calls each third_party trainer; artifact keeps the old lightweight checkpoint contract.",
    )
    parser.add_argument(
        "--global-prompt-artifact",
        default=None,
        help=(
            "Optional server prompt artifact from a previous round. Its soft_prompt "
            "tensor is injected as virtual token embeddings into causal LM calls."
        ),
    )
    parser.add_argument("--max-train-samples-per-client", type=int, default=None)
    parser.add_argument(
        "--max-valid-samples-per-client",
        type=int,
        default=None,
        help="Maximum validation examples per client for round accuracy.",
    )
    parser.add_argument(
        "--max-samples-per-client",
        type=int,
        default=None,
        help=(
            "Override data.max_trajectory_samples_per_client for held-out "
            "summary trajectories in post-training rounds."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--summary-max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    run_name, run_root = resolve_run_root(args)
    config_path = (
        Path(args.config)
        if args.config
        else run_root / "config.yaml"
        if args.start_round > 0 and (run_root / "config.yaml").exists()
        else Path("configs/fedpao_six_clients.yaml")
    )
    config = read_config(config_path)
    resolve_runtime_args(args, config)
    if args.start_round > args.rounds:
        raise ValueError(
            f"--start-round {args.start_round} cannot be greater than --rounds {args.rounds}"
        )

    run_root.mkdir(parents=True, exist_ok=True)
    write_yaml(run_root / "config.yaml", config)

    clients = selected_clients(config, args.mas)
    completed_rounds = []
    if args.start_round <= 0:
        backend = load_backend(args.model_path, args.dtype) if args.round0_trainer == "native" else None
        previous_manifest = round0_train(
            args,
            config,
            clients,
            run_root,
            args.max_train_samples_per_client,
            backend,
        )
        completed_rounds.append(
            {
                "round": 0,
                "manifest": str(run_root / "round_000" / "manifest.json"),
            }
        )
        first_post_training_round = 1
    else:
        previous_round = args.start_round - 1
        previous_manifest = (
            load_round0_manifest_for_run(run_root)
            if previous_round == 0
            else load_round_manifest_for_run(run_root, previous_round)
        )
        print(
            "loaded previous round manifest: "
            f"{run_root / f'round_{previous_round:03d}' / 'manifest.json'}"
        )
        first_post_training_round = args.start_round

    if args.rounds < 1:
        print(f"round0 complete: {run_root / 'round_000' / 'manifest.json'}")
        return

    previous_global_prompt_path = args.global_prompt_artifact
    if args.start_round > 1:
        previous_global = previous_manifest.get("global_prompt") or {}
        previous_global_prompt_path = previous_global.get("global_prompt_artifact_path")

    for round_index in range(first_post_training_round, args.rounds + 1):
        round_manifest = run_federated_round(
            args,
            config,
            clients,
            run_root,
            previous_manifest,
            round_index,
            previous_global_prompt_path,
        )
        completed_rounds.append(
            {
                "round": round_index,
                "manifest": str(run_root / f"round_{round_index:03d}" / "manifest.json"),
                "global_prompt_artifact": (
                    (round_manifest.get("global_prompt") or {}).get(
                        "global_prompt_artifact_path"
                    )
                ),
            }
        )
        previous_manifest = round_manifest
        previous_global_prompt_path = (
            (round_manifest.get("global_prompt") or {}).get(
                "global_prompt_artifact_path"
            )
        )

    write_json(
        run_root / "manifest.json",
        {
            "run_name": run_name,
            "run_root": str(run_root),
            "config": str(config_path),
            "start_round": args.start_round,
            "rounds": args.rounds,
            "round0_manifest": str(run_root / "round_000" / "manifest.json"),
            "completed_rounds": completed_rounds,
        },
    )


if __name__ == "__main__":
    main()
