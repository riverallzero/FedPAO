from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_partitioning import (  # noqa: E402
    client_partition_indices,
    client_partition_metadata,
    dataset_partition_key,
    partition_mode,
    shuffled_dataset_indices,
)
from scripts.run import (  # noqa: E402
    filter_third_party_execution_noise,
    generate_with_embedding_soft_prompt,
    load_global_prompt_state,
    rebase_round0_manifest_paths,
)
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


DEFAULT_MAX_NEW_TOKENS = 1024


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


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


def full_benchmark_dataset(client: dict[str, Any]):
    dataset = load_dataset(client["dataset_name"])
    splits = [dataset[name] for name in dataset.keys()]
    if len(splits) == 1:
        return splits[0]
    return concatenate_datasets(splits)


def used_round0_indices(
    *,
    total_size: int,
    client: dict[str, Any],
    clients: list[dict[str, Any]],
    seed: int,
    train_fraction: float,
    sample_limit: int | None,
    mode: str,
) -> set[int]:
    indices = client_partition_indices(total_size, client, clients, seed, sample_limit, mode)
    train_size = int(len(indices) * train_fraction)
    train_size = (
        min(max(train_size, 1), max(len(indices) - 1, 1))
        if len(indices) > 1
        else len(indices)
    )
    return set(indices[:train_size]) | set(indices[train_size:])


def used_post_round_indices(
    *,
    total_size: int,
    client: dict[str, Any],
    clients: list[dict[str, Any]],
    seed: int,
    train_fraction: float,
    sample_limit: int | None,
    trajectory_limit: int | None,
    mode: str,
) -> set[int]:
    partition_limit = sample_limit
    if sample_limit is not None and trajectory_limit is not None:
        partition_limit = sample_limit + trajectory_limit
    indices = client_partition_indices(
        total_size,
        client,
        clients,
        seed,
        partition_limit,
        mode,
    )

    train_valid_indices = indices[:sample_limit] if sample_limit is not None else indices
    train_size = int(len(train_valid_indices) * train_fraction)
    train_size = (
        min(max(train_size, 1), max(len(train_valid_indices) - 1, 1))
        if len(train_valid_indices) > 1
        else len(train_valid_indices)
    )
    train_indices = train_valid_indices[:train_size]
    valid_indices = train_valid_indices[train_size:]
    trajectory_indices = indices[sample_limit:] if sample_limit is not None else []
    if not trajectory_indices:
        used_train_valid = set(train_indices) | set(valid_indices)
        trajectory_indices = [index for index in indices if index not in used_train_valid]
    if trajectory_limit is not None:
        trajectory_indices = trajectory_indices[:trajectory_limit]
    return set(train_indices) | set(valid_indices) | set(trajectory_indices)


def shared_heldout_test_indices_by_dataset(
    *,
    datasets_by_client: dict[str, Any],
    clients: list[dict[str, Any]],
    config: dict[str, Any],
    n: int,
) -> dict[str, list[int]]:
    data_cfg = config.get("data", {})
    mode = partition_mode(data_cfg)
    seed = int(config.get("seed", 42))
    train_fraction = float(data_cfg.get("train_fraction", 0.8))
    sample_limit_value = data_cfg.get("max_samples_per_client")
    sample_limit = int(sample_limit_value) if sample_limit_value is not None else None
    trajectory_limit_value = data_cfg.get("max_trajectory_samples_per_client", 3)
    trajectory_limit = (
        int(trajectory_limit_value) if trajectory_limit_value is not None else None
    )

    grouped_clients: dict[str, list[dict[str, Any]]] = {}
    for client in clients:
        grouped_clients.setdefault(dataset_partition_key(client), []).append(client)

    indices_by_key: dict[str, list[int]] = {}
    for key, peers in grouped_clients.items():
        representative = peers[0]
        total_size = len(datasets_by_client[representative["id"]])
        used: set[int] = set()
        for peer in peers:
            peer_total_size = len(datasets_by_client[peer["id"]])
            if peer_total_size != total_size:
                raise ValueError(
                    f"Clients sharing dataset key {key!r} resolved to different "
                    f"dataset sizes: {representative['id']}={total_size}, "
                    f"{peer['id']}={peer_total_size}"
                )
            used |= used_round0_indices(
                total_size=total_size,
                client=peer,
                clients=clients,
                seed=seed,
                train_fraction=train_fraction,
                sample_limit=sample_limit,
                mode=mode,
            )
            used |= used_post_round_indices(
                total_size=total_size,
                client=peer,
                clients=clients,
                seed=seed,
                train_fraction=train_fraction,
                sample_limit=sample_limit,
                trajectory_limit=trajectory_limit,
                mode=mode,
            )

        candidates = [
            index
            for index in shuffled_dataset_indices(total_size, representative, seed)
            if index not in used
        ]
        if len(candidates) < n:
            peer_ids = ", ".join(peer["id"] for peer in peers)
            raise RuntimeError(
                f"Only {len(candidates)} shared held-out test examples remain for "
                f"dataset {key!r} ({peer_ids}) after excluding all peer "
                f"train/validation/trajectory indices; requested {n}."
            )
        indices_by_key[key] = candidates[:n]

    return {
        client["id"]: indices_by_key[dataset_partition_key(client)]
        for client in clients
    }


def load_round_manifest(run_root: Path, round_index: int) -> dict[str, Any]:
    manifest_path = run_root / f"round_{round_index:03d}" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing round manifest: {manifest_path}")
    return rebase_round0_manifest_paths(read_json(manifest_path), run_root)


def discover_rounds(run_root: Path) -> list[int]:
    rounds = []
    for path in sorted(run_root.glob("round_[0-9][0-9][0-9]/manifest.json")):
        try:
            rounds.append(int(path.parent.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not rounds:
        raise FileNotFoundError(f"No round manifests found under {run_root}")
    return rounds


def client_records_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["client_id"]: record for record in manifest.get("clients", [])}


def llm_generate_for_round(
    *,
    backend: dict[str, Any],
    config: dict[str, Any],
    max_new_tokens: int,
    prompt_artifact_path: str | None,
):
    prompt_state = load_global_prompt_state(prompt_artifact_path)
    if prompt_state is None:
        return lambda prompt: generate(backend, prompt, max_new_tokens), False
    return (
        lambda prompt: generate_with_embedding_soft_prompt(
            backend,
            prompt,
            max_new_tokens,
            prompt_state,
            config,
        ),
        True,
    )


def prompt_artifact_for_round(
    manifest: dict[str, Any],
    prompt_mode: str,
) -> str | None:
    if prompt_mode == "none":
        return None
    global_prompt = manifest.get("global_prompt") or {}
    path = global_prompt.get("global_prompt_artifact_path")
    if prompt_mode == "manifest":
        return path
    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def resolve_max_new_tokens(config: dict[str, Any]) -> int:
    evaluation_cfg = config.get("evaluation", {})
    generation_cfg = config.get("generation", {})
    return int(
        evaluation_cfg.get(
            "max_new_tokens",
            generation_cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
        )
    )


def write_client_test_set(
    *,
    output_dir: Path,
    client: dict[str, Any],
    dataset,
    test_indices: list[int],
) -> str:
    test_set_dir = output_dir / "test_sets"
    test_set_dir.mkdir(parents=True, exist_ok=True)
    path = test_set_dir / f"{client['id']}_{client['mas']}_{client['dataset']}.jsonl"
    examples = dataset.select(test_indices)
    with path.open("w", encoding="utf-8") as file:
        for local_index, example in enumerate(examples):
            task_prompt, ground_truth = make_prompt(client, dict(example))
            row = {
                "client_id": client["id"],
                "mas": client["mas"],
                "dataset": client["dataset"],
                "dataset_name": client["dataset_name"],
                "example_index": local_index,
                "source_example_index": test_indices[local_index],
                "input": task_prompt,
                "ground_truth": ground_truth,
                "example": dict(example),
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def evaluate_client_round(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    client: dict[str, Any],
    clients: list[dict[str, Any]],
    record: dict[str, Any],
    round_index: int,
    dataset,
    test_indices: list[int],
    backend: dict[str, Any],
    max_new_tokens: int,
    prompt_artifact_path: str | None,
    output_file,
) -> dict[str, Any]:
    llm_generate, prompt_enabled = llm_generate_for_round(
        backend=backend,
        config=config,
        max_new_tokens=max_new_tokens,
        prompt_artifact_path=prompt_artifact_path,
    )
    orchestrator = create_orchestrator(
        client["mas"],
        third_party_dir=args.third_party_dir,
        mode=args.orchestrator_mode,
        round0_checkpoint_path=record["checkpoint_path"],
    )

    rows = []
    examples = dataset.select(test_indices)
    data_partition_mode = partition_mode(config.get("data", {}))
    for local_index, example in enumerate(examples):
        task_prompt, ground_truth = make_prompt(client, dict(example))
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
        if printable_stdout:
            print(printable_stdout, end="")
        if printable_stderr:
            print(printable_stderr, end="", file=sys.stderr)

        response = orchestrated.metadata.get("response", "")
        prediction = normalize_prediction(client["dataset"], response)
        correct = is_correct(client["dataset"], prediction, ground_truth)
        row = {
            "round": round_index,
            "client_id": client["id"],
            "mas": client["mas"],
            "dataset": client["dataset"],
            "dataset_name": client["dataset_name"],
            **client_partition_metadata(client, clients, data_partition_mode),
            "role": "test",
            "example_index": local_index,
            "source_example_index": test_indices[local_index],
            "model_path": args.model_path,
            "checkpoint_path": record["checkpoint_path"],
            "hydra_run_dir": record.get("hydra_run_dir"),
            "orchestrator_mode": args.orchestrator_mode,
            "global_prompt_artifact": prompt_artifact_path,
            "embedding_soft_prompt_enabled": prompt_enabled,
            "suppressed_execution_errors": (
                suppressed_stdout_errors + suppressed_stderr_errors
            ),
            "orchestrator_metadata": orchestrated.metadata,
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
            f"round{round_index} test {client['id']} {client['mas']} {client['dataset']}: "
            f"pred={prediction!r} gt={ground_truth!r} "
            f"correct={correct} latency={row['latency_sec']}s"
        )

    correct_count = sum(row["correct"] for row in rows)
    return {
        "round": round_index,
        "client_id": client["id"],
        "mas": client["mas"],
        "dataset": client["dataset"],
        "dataset_name": client["dataset_name"],
        "checkpoint_path": record["checkpoint_path"],
        "global_prompt_artifact": prompt_artifact_path,
        "embedding_soft_prompt_enabled": prompt_enabled,
        "num_examples": len(rows),
        "correct": correct_count,
        "accuracy": correct_count / len(rows) if rows else None,
        "test_indices": test_indices,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved FedPAO round checkpoints on per-client held-out test "
            "examples that are excluded from train/validation/trajectory data."
        )
    )
    parser.add_argument("--run-root", required=True, help="Existing FedPAO run directory.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-path")
    parser.add_argument("--third-party-dir", default="third_party")
    parser.add_argument("--mas", choices=SUPPORTED_MAS, nargs="+")
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=None,
        help="Round numbers to evaluate. Defaults to every round manifest under --run-root.",
    )
    parser.add_argument(
        "--test-samples-per-client",
        type=int,
        default=20,
        help="Number of held-out test examples per client.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for test outputs. Defaults to <run-root>/test.",
    )
    parser.add_argument(
        "--orchestrator-mode",
        choices=["native", "simulated"],
        default="native",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["manifest", "none"],
        default="manifest",
        help=(
            "manifest injects a round manifest's global prompt when present; "
            "none evaluates checkpoints without embedding soft prompts."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    run_root = Path(args.run_root).resolve()
    config_path = (
        Path(args.config)
        if args.config
        else run_root / "config.yaml"
        if (run_root / "config.yaml").exists()
        else Path("configs/fedpao_six_clients.yaml")
    )
    config = read_yaml(config_path)
    if args.model_path is None:
        args.model_path = config.get("model", {}).get("model_id", DEFAULT_MODEL_PATH)
    max_new_tokens = resolve_max_new_tokens(config)

    clients = selected_clients(config, args.mas)
    rounds = args.rounds if args.rounds is not None else discover_rounds(run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "test_inference.jsonl"
    manifest_path = output_dir / "test_manifest.json"

    backend = load_backend(args.model_path, args.dtype)
    datasets_by_client = {
        client["id"]: full_benchmark_dataset(client)
        for client in clients
    }
    test_indices_by_client = shared_heldout_test_indices_by_dataset(
        datasets_by_client=datasets_by_client,
        clients=clients,
        config=config,
        n=args.test_samples_per_client,
    )
    test_set_paths = {
        client["id"]: write_client_test_set(
            output_dir=output_dir,
            client=client,
            dataset=datasets_by_client[client["id"]],
            test_indices=test_indices_by_client[client["id"]],
        )
        for client in clients
    }

    summaries = []
    with jsonl_path.open("w", encoding="utf-8") as out:
        for round_index in rounds:
            manifest = load_round_manifest(run_root, round_index)
            records = client_records_by_id(manifest)
            prompt_artifact_path = prompt_artifact_for_round(manifest, args.prompt_mode)
            for client in clients:
                if client["id"] not in records:
                    raise KeyError(
                        f"Round {round_index} manifest has no record for {client['id']}"
                    )
                summary = evaluate_client_round(
                    args=args,
                    config=config,
                    client=client,
                    clients=clients,
                    record=records[client["id"]],
                    round_index=round_index,
                    dataset=datasets_by_client[client["id"]],
                    test_indices=test_indices_by_client[client["id"]],
                    backend=backend,
                    max_new_tokens=max_new_tokens,
                    prompt_artifact_path=prompt_artifact_path,
                    output_file=out,
                )
                summaries.append(summary)

    total_correct = sum(item["correct"] for item in summaries)
    total_examples = sum(item["num_examples"] for item in summaries)
    result = {
        "run_root": str(run_root),
        "config": str(config_path),
        "model_path": args.model_path,
        "max_new_tokens": max_new_tokens,
        "orchestrator_mode": args.orchestrator_mode,
        "prompt_mode": args.prompt_mode,
        "rounds": rounds,
        "test_samples_per_client": args.test_samples_per_client,
        "inference_path": str(jsonl_path),
        "test_sets": test_set_paths,
        "clients": summaries,
        "accuracy": {
            "correct": total_correct,
            "num_examples": total_examples,
            "value": total_correct / total_examples if total_examples else None,
        },
    }
    write_json(manifest_path, result)
    print(f"wrote: {jsonl_path}")
    print(f"manifest: {manifest_path}")
    print(f"accuracy: {total_correct}/{total_examples}")


if __name__ == "__main__":
    main()
