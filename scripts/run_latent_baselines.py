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


def evaluate_baseline(
    baseline_name: str,
    client_config: dict,
    local_client: LocalClient,
    eval_examples: list,
    latent: np.ndarray | None,
    extra: dict | None = None,
) -> dict:
    summary, _ = local_client.evaluate(eval_examples, latent)
    return {
        "baseline": baseline_name,
        "client_id": client_config["id"],
        "mas": client_config["mas"],
        "dataset": client_config["dataset"],
        **summary,
        **(extra or {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedpao_smoke_taskonly.yaml")
    parser.add_argument("--output-dir", default="outputs/latent_baselines")
    parser.add_argument("--clients-per-round", type=int)
    parser.add_argument("--max-train-samples-per-client", type=int, default=1)
    parser.add_argument("--max-eval-samples-per-client", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=2)
    parser.add_argument("--local-iterations", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--include-local-cma", action="store_true")
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
    latent_dim = int(prompt_config.get("latent_dim", 32))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "baseline_metrics.jsonl"

    backend = HFCausalBackend(config["model"]["model_id"], dtype=args.dtype)
    projector = SoftPromptProjector(
        latent_dim=latent_dim,
        prompt_length=int(prompt_config.get("prompt_length", 8)),
        hidden_size=backend.hidden_size,
        init_std=float(prompt_config.get("init_std", 0.02)),
        seed=int(prompt_config.get("projection_seed", config.get("seed", 42))),
    )
    metadata = {
        "model_id": config["model"]["model_id"],
        "latent_dim": latent_dim,
        "prompt_length": int(prompt_config.get("prompt_length", 8)),
        "hidden_size": backend.hidden_size,
        "projection_seed": int(prompt_config.get("projection_seed", config.get("seed", 42))),
        "projection_hash": projector.projection_hash(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )

    rng = np.random.default_rng(int(config.get("seed", 42)))
    selected_clients = config["clients"][: args.clients_per_round or len(config["clients"])]

    with metrics_path.open("w") as out:
        for client_config in selected_clients:
            orchestrator = MockOrchestrator(
                client_config,
                backend,
                projector,
                max_new_tokens=args.max_new_tokens,
            )
            local_client = LocalClient(client_config, orchestrator)
            eval_examples = load_client_examples(
                config,
                client_config,
                "eval",
                args.max_eval_samples_per_client,
            )

            zero_latent = np.zeros(latent_dim, dtype=np.float32)
            random_latent = rng.normal(0.0, 1.0, size=latent_dim).astype(np.float32)
            rows = [
                evaluate_baseline(
                    "no_soft_prompt",
                    client_config,
                    local_client,
                    eval_examples,
                    None,
                ),
                evaluate_baseline(
                    "zero_latent",
                    client_config,
                    local_client,
                    eval_examples,
                    zero_latent,
                    {"z_norm": float(np.linalg.norm(zero_latent))},
                ),
                evaluate_baseline(
                    "random_latent",
                    client_config,
                    local_client,
                    eval_examples,
                    random_latent,
                    {"z_norm": float(np.linalg.norm(random_latent))},
                ),
            ]

            if args.include_local_cma:
                train_examples = load_client_examples(
                    config,
                    client_config,
                    "train",
                    args.max_train_samples_per_client,
                )

                def objective(latent: np.ndarray):
                    return local_client.evaluate_loss(
                        train_examples,
                        latent,
                        config.get("objective", {}),
                    )

                optimizer = CmaEsOptimizer(
                    sigma0=float(optimizer_config.get("sigma0", 0.5)),
                    population_size=args.population_size,
                    local_iterations=args.local_iterations,
                    seed=int(config.get("seed", 42)),
                )
                result = optimizer.optimize(zero_latent, objective)
                rows.append(
                    evaluate_baseline(
                        "local_only_cma",
                        client_config,
                        local_client,
                        eval_examples,
                        result.best_latent,
                        {
                            "z_norm": float(np.linalg.norm(result.best_latent)),
                            "train_loss": result.best_loss,
                            "num_evaluations": result.num_evaluations,
                        },
                    )
                )

            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"{client_config['id']} {client_config['mas']} {client_config['dataset']}: "
                f"wrote {len(rows)} baselines"
            )

    print(f"wrote: {metrics_path}")


if __name__ == "__main__":
    main()
