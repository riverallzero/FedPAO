import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

from src.clients.local_client import LocalClient
from src.data.loader import load_client_examples
from src.models.hf_causal import HFCausalBackend
from src.optimizers.cma_es_optimizer import CmaEsOptimizer
from src.orchestrators.mock import MockOrchestrator
from src.prompts.soft_prompt import SoftPromptProjector
from src.server.fedavg import communication_bytes, fedavg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedpao_six_clients.yaml")
    parser.add_argument("--output-dir", default="outputs/fedpao_dryrun")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--clients-per-round", type=int)
    parser.add_argument("--max-train-samples-per-client", type=int, default=1)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--local-iterations", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    prompt_config = config.get("prompt", {})
    optimizer_config = config.get("optimizer", {})
    federated_config = config.get("federated", {})
    latent_dim = int(prompt_config.get("latent_dim", 32))
    rounds = args.rounds or int(federated_config.get("rounds", 10))
    clients_per_round = args.clients_per_round or int(
        federated_config.get("clients_per_round", len(config["clients"]))
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    client_metrics_path = output_dir / "client_metrics.jsonl"

    backend = HFCausalBackend(config["model"]["model_id"], dtype=args.dtype)
    projector = SoftPromptProjector(
        latent_dim=latent_dim,
        prompt_length=int(prompt_config.get("prompt_length", 8)),
        hidden_size=backend.hidden_size,
        init_std=float(prompt_config.get("init_std", 0.02)),
        seed=int(prompt_config.get("projection_seed", config.get("seed", 42))),
    )

    global_latent = np.zeros(latent_dim, dtype=np.float32)
    with metrics_path.open("w") as metrics_out, client_metrics_path.open("w") as client_out:
        for round_index in range(rounds):
            selected_clients = config["clients"][:clients_per_round]
            local_latents = []
            weights = []

            round_dir = output_dir / f"round_{round_index:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            np.save(round_dir / "global_latent_input.npy", global_latent)

            for client_index, client_config in enumerate(selected_clients):
                examples = load_client_examples(
                    config,
                    client_config,
                    "train",
                    args.max_train_samples_per_client,
                )
                orchestrator = MockOrchestrator(
                    client_config,
                    backend,
                    projector,
                    max_new_tokens=args.max_new_tokens,
                )
                local_client = LocalClient(client_config, orchestrator)

                def objective(latent: np.ndarray):
                    return local_client.evaluate_loss(
                        examples,
                        latent,
                        config.get("objective", {}),
                    )

                optimizer = CmaEsOptimizer(
                    sigma0=float(optimizer_config.get("sigma0", 0.5)),
                    population_size=args.population_size
                    or int(optimizer_config.get("population_size", 8)),
                    local_iterations=args.local_iterations
                    or int(optimizer_config.get("local_iterations", 5)),
                    seed=int(config.get("seed", 42)) + round_index * 1000 + client_index,
                )
                result = optimizer.optimize(global_latent, objective)
                local_latents.append(result.best_latent)
                weights.append(len(examples))

                client_row = {
                    "round": round_index,
                    "client_id": client_config["id"],
                    "num_examples": len(examples),
                    "local_loss": result.best_loss,
                    "local_accuracy": result.best_metrics.get("accuracy", 0.0),
                    "z_norm": float(np.linalg.norm(result.best_latent)),
                    "communication_bytes": communication_bytes(result.best_latent),
                    "num_evaluations": result.num_evaluations,
                }
                client_out.write(json.dumps(client_row, ensure_ascii=False) + "\n")
                client_out.flush()
                print(
                    f"round={round_index} client={client_config['id']} "
                    f"loss={result.best_loss:.4f} "
                    f"acc={client_row['local_accuracy']:.3f}"
                )

            if federated_config.get("weight_by") == "num_examples":
                global_latent = fedavg(local_latents, weights)
            else:
                global_latent = fedavg(local_latents)

            np.save(round_dir / "global_latent_output.npy", global_latent)
            round_row = {
                "round": round_index,
                "num_clients": len(selected_clients),
                "global_z_norm": float(np.linalg.norm(global_latent)),
                "communication_bytes": communication_bytes(global_latent)
                * len(selected_clients),
            }
            metrics_out.write(json.dumps(round_row, ensure_ascii=False) + "\n")
            metrics_out.flush()

    print(f"wrote: {metrics_path}")
    print(f"wrote: {client_metrics_path}")


if __name__ == "__main__":
    main()
