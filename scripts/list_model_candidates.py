import argparse
import json
from pathlib import Path


TEXT_MODEL_TYPES = {
    "qwen2",
    "qwen3",
    "qwen3_5",
    "qwen3_5_text",
    "llama",
    "mistral",
    "gemma",
    "phi3",
}


def is_causal_candidate(config: dict) -> bool:
    architecture = ",".join(config.get("architectures") or [])
    text_architecture = ",".join((config.get("text_config") or {}).get("architectures") or [])
    model_type = str(config.get("model_type"))
    return (
        "ForCausalLM" in architecture
        or "CausalLM" in architecture
        or "ForCausalLM" in text_architecture
        or "CausalLM" in text_architecture
        or model_type in TEXT_MODEL_TYPES
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/digirl/checkpoints")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    rows = []
    for path in Path(args.root).rglob("config.json"):
        try:
            config = json.loads(path.read_text())
        except Exception:
            continue
        if not is_causal_candidate(config):
            continue
        rows.append(
            {
                "path": str(path.parent),
                "model_type": config.get("model_type"),
                "architectures": config.get("architectures"),
                "text_architectures": (config.get("text_config") or {}).get("architectures"),
            }
        )

    for row in rows[: args.limit]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"total_candidates={len(rows)}")


if __name__ == "__main__":
    main()
