from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from src.data.loader import load_client_examples
from src.models.hf_causal import HFCausalBackend
from src.orchestrators.registry import build_orchestrator
from src.prompts.soft_prompt import SoftPromptProjector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedpao_six_clients.yaml")
    parser.add_argument("--mas", choices=["puppeteer", "masrouter", "gptswarm"], required=True)
    parser.add_argument("--client-id")
    parser.add_argument("--split-role", choices=["train", "eval", "test"], default="train")
    parser.add_argument("--output", default="outputs/native_orchestrator_soft_prompt_smoke.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--latent", choices=["zero", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    candidates = [
        client
        for client in config["clients"]
        if client["mas"] == args.mas
        and (args.client_id is None or client["id"] == args.client_id)
    ]
    if not candidates:
        raise ValueError(f"No client matched mas={args.mas!r} client_id={args.client_id!r}")
    client_config = candidates[0]

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
    if args.latent == "zero":
        latent = np.zeros(latent_dim, dtype=np.float32)
    else:
        rng = np.random.default_rng(args.seed)
        latent = rng.normal(size=latent_dim).astype(np.float32)

    examples = load_client_examples(config, client_config, args.split_role, max_samples=1)
    if not examples:
        raise RuntimeError("No examples loaded for smoke test")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    orchestrator = build_orchestrator(
        "native",
        client_config,
        backend,
        projector,
        max_new_tokens=args.max_new_tokens,
        config=config,
    )

    row = {
        "client_id": client_config["id"],
        "mas": client_config["mas"],
        "dataset": client_config["dataset"],
        "task_id": examples[0].task_id,
        "split": examples[0].split,
        "latent": args.latent,
        "latent_norm": float(np.linalg.norm(latent)),
    }
    try:
        result = orchestrator.run(examples[0], latent)
        row.update(
            {
                "ok": True,
                "final_answer": result.final_answer,
                "metrics": result.metrics,
                "metadata": result.metadata,
            }
        )
    except Exception as exc:  # noqa: BLE001 - smoke script records adapter failures.
        row.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    with output_path.open("w") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, indent=2, ensure_ascii=False))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
