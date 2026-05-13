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
from src.orchestrators.mock import MockOrchestrator
from src.prompts.soft_prompt import SoftPromptProjector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedpao_six_clients.yaml")
    parser.add_argument("--output-dir", default="outputs/mock_orchestrator")
    parser.add_argument("--split-role", choices=["train", "eval", "test"], default="eval")
    parser.add_argument("--max-samples-per-client", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client_metrics_path = output_dir / "client_metrics.jsonl"
    prediction_path = output_dir / "predictions.jsonl"

    backend = HFCausalBackend(config["model"]["model_id"], dtype=args.dtype)
    prompt_config = config.get("prompt", {})
    projector = SoftPromptProjector(
        latent_dim=int(prompt_config.get("latent_dim", 32)),
        prompt_length=int(prompt_config.get("prompt_length", 8)),
        hidden_size=backend.hidden_size,
        init_std=float(prompt_config.get("init_std", 0.02)),
        seed=int(prompt_config.get("projection_seed", config.get("seed", 42))),
    )
    latent = np.zeros(int(prompt_config.get("latent_dim", 32)), dtype=np.float32)

    with client_metrics_path.open("w") as metrics_out, prediction_path.open("w") as pred_out:
        for client_config in config["clients"]:
            examples = load_client_examples(
                config,
                client_config,
                args.split_role,
                args.max_samples_per_client,
            )
            orchestrator = MockOrchestrator(
                client_config,
                backend,
                projector,
                max_new_tokens=args.max_new_tokens,
            )
            local_client = LocalClient(client_config, orchestrator)
            summary, rows = local_client.evaluate(examples, latent)
            metrics_out.write(json.dumps(summary, ensure_ascii=False) + "\n")
            metrics_out.flush()
            for row in rows:
                pred_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            pred_out.flush()
            print(
                f"{summary['client_id']} {summary['mas']} {summary['dataset']}: "
                f"acc={summary['accuracy']:.3f} "
                f"avg_tokens={summary['avg_total_tokens']:.1f}"
            )

    print(f"wrote: {client_metrics_path}")
    print(f"wrote: {prediction_path}")


if __name__ == "__main__":
    main()
