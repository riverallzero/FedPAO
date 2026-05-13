import argparse
import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

from src.models.hf_causal import HFCausalBackend
from src.prompts.soft_prompt import SoftPromptProjector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedpao_six_clients.yaml")
    parser.add_argument("--model-path")
    parser.add_argument("--prompt", default="Return only the letter A.")
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    prompt_config = config.get("prompt", {})
    model_path = args.model_path or config["model"]["model_id"]
    latent_dim = args.latent_dim or int(prompt_config.get("latent_dim", 32))
    prompt_length = args.prompt_length or int(prompt_config.get("prompt_length", 8))

    backend = HFCausalBackend(model_path, dtype=args.dtype)
    projector = SoftPromptProjector(
        latent_dim=latent_dim,
        prompt_length=prompt_length,
        hidden_size=backend.hidden_size,
        init_std=float(prompt_config.get("init_std", 0.02)),
        seed=int(prompt_config.get("projection_seed", config.get("seed", 42))),
    )
    latent = np.zeros(latent_dim, dtype=np.float32)
    soft_prompt = projector.project(latent, device=backend.device)

    base = backend.generate(args.prompt, args.max_new_tokens)
    soft = backend.generate_with_soft_prompt(args.prompt, soft_prompt, args.max_new_tokens)

    frozen = all(not parameter.requires_grad for parameter in backend.model.parameters())
    print(f"model_class: {type(backend.model).__name__}")
    print(f"hidden_size: {backend.hidden_size}")
    print(f"soft_prompt_shape: {tuple(soft_prompt.shape)}")
    print(f"projection_hash: {projector.projection_hash()}")
    print(f"parameters_frozen: {frozen}")
    print(f"base_generation: {base.text!r}")
    print(f"soft_generation: {soft.text!r}")
    print(f"base_tokens: {base.total_tokens}")
    print(f"soft_tokens: {soft.total_tokens}")


if __name__ == "__main__":
    main()
