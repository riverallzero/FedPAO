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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedpao_six_clients.yaml")
    parser.add_argument("--client-id", default="c1")
    parser.add_argument("--output-dir", default="outputs/local_cma_smoke")
    parser.add_argument("--split-role", choices=["train", "eval", "test"], default="train")
    parser.add_argument("--max-train-samples", type=int, default=1)
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

    client_config = next(
        client for client in config["clients"] if client["id"] == args.client_id
    )
    optimizer_config = config.get("optimizer", {})
    prompt_config = config.get("prompt", {})
    latent_dim = int(prompt_config.get("latent_dim", 32))

    backend = HFCausalBackend(config["model"]["model_id"], dtype=args.dtype)
    projector = SoftPromptProjector(
        latent_dim=latent_dim,
        prompt_length=int(prompt_config.get("prompt_length", 8)),
        hidden_size=backend.hidden_size,
        init_std=float(prompt_config.get("init_std", 0.02)),
        seed=int(prompt_config.get("projection_seed", config.get("seed", 42))),
    )
    orchestrator = MockOrchestrator(
        client_config,
        backend,
        projector,
        max_new_tokens=args.max_new_tokens,
    )
    local_client = LocalClient(client_config, orchestrator)
    examples = load_client_examples(
        config,
        client_config,
        args.split_role,
        args.max_train_samples,
    )
    initial_latent = np.zeros(latent_dim, dtype=np.float32)

    def objective(latent: np.ndarray):
        return local_client.evaluate_loss(examples, latent, config.get("objective", {}))

    initial_loss, initial_metrics = objective(initial_latent)
    optimizer = CmaEsOptimizer(
        sigma0=float(optimizer_config.get("sigma0", 0.5)),
        population_size=args.population_size
        or int(optimizer_config.get("population_size", 8)),
        local_iterations=args.local_iterations
        or int(optimizer_config.get("local_iterations", 5)),
        seed=int(config.get("seed", 42)),
    )
    result = optimizer.optimize(initial_latent, objective)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "best_latent.npy", result.best_latent)
    metrics = {
        "client_id": client_config["id"],
        "latent_dim": latent_dim,
        "prompt_length": int(prompt_config.get("prompt_length", 8)),
        "hidden_size": backend.hidden_size,
        "projection_seed": int(prompt_config.get("projection_seed", config.get("seed", 42))),
        "projection_hash": projector.projection_hash(),
        "initial_loss": initial_loss,
        "initial_metrics": initial_metrics,
        "best_loss": result.best_loss,
        "best_metrics": result.best_metrics,
        "num_evaluations": result.num_evaluations,
        "history": result.history,
        "best_z_norm": float(np.linalg.norm(result.best_latent)),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
