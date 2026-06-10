import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestrators import create_orchestrator
from scripts.data_partitioning import (
    client_partition_indices,
    client_partition_metadata,
    partition_mode,
)


DEFAULT_MODEL_PATH = "/data/digirl/checkpoints/llava-onevision-qwen2-0.5b-ov-hf"
SUPPORTED_MAS = ("masrouter", "gptswarm")


def load_backend(model_path: str, dtype: str):
    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch_dtype,
            device_map="auto",
        )
        return {"kind": "image_text", "model": model.eval(), "processor": processor}
    except Exception as image_text_error:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch_dtype,
            device_map="auto",
        )
        return {
            "kind": "causal_lm",
            "model": model.eval(),
            "tokenizer": tokenizer,
            "image_text_error": repr(image_text_error),
        }


def make_prompt(client: dict, example: dict) -> tuple[str, str]:
    mas = client["mas"]
    dataset = client["dataset"]

    if dataset == "mmlu_pro":
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options = "\n".join(
            f"{letters[idx]}. {option}" for idx, option in enumerate(example["options"])
        )
        prompt = (
            f"You are the local frozen LLM backend for the {mas} client.\n"
            "Answer the multiple-choice question. Return only the option letter.\n\n"
            f"Question: {example['question']}\n"
            f"Options:\n{options}\n"
            "Answer:"
        )
        return prompt, str(example["answer"]).strip()

    if dataset == "gsm_hard":
        prompt = (
            f"You are the local frozen LLM backend for the {mas} client.\n"
            "Solve the math problem. Return only the final numeric answer.\n\n"
            f"Problem: {example['input']}\n"
            "Answer:"
        )
        return prompt, str(example["target"]).strip()

    raise ValueError(f"Unsupported dataset: {dataset}")


def format_chat(processor, prompt: str) -> str:
    if not hasattr(processor, "apply_chat_template"):
        return prompt

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def first_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(backend: dict, prompt: str, max_new_tokens: int) -> str:
    model = backend["model"]

    if backend["kind"] == "image_text":
        processor = backend["processor"]
        text = format_chat(processor, prompt)
        inputs = processor(text=text, return_tensors="pt")
        inputs = {key: value.to(first_model_device(model)) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return processor.decode(output[0][input_len:], skip_special_tokens=True).strip()

    tokenizer = backend["tokenizer"]
    inputs = tokenizer(prompt, return_tensors="pt").to(first_model_device(model))
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()


def normalize_prediction(dataset: str, text: str) -> str:
    if dataset == "mmlu_pro":
        match = re.search(r"\b([A-J])\b", text.upper())
        return match.group(1) if match else text.strip()

    if dataset == "gsm_hard":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return numbers[-1] if numbers else text.strip()

    return text.strip()


def is_correct(dataset: str, prediction: str, ground_truth: str) -> bool:
    if dataset == "gsm_hard":
        try:
            return float(prediction) == float(ground_truth)
        except ValueError:
            return False
    return prediction == ground_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedpao_six_clients.yaml")
    parser.add_argument("--model-path")
    parser.add_argument("--output", default="outputs/client_inference_smoke.jsonl")
    parser.add_argument("--third-party-dir", default="third_party")
    parser.add_argument(
        "--mas",
        choices=SUPPORTED_MAS,
        nargs="+",
        help="Run only clients for the selected orchestrator(s).",
    )
    parser.add_argument(
        "--orchestrator-mode",
        choices=["native", "simulated"],
        default="native",
        help="native runs the cloned baseline framework; simulated only wraps the prompt.",
    )
    parser.add_argument(
        "--data-partition-mode",
        choices=["shared", "client_disjoint"],
        default=None,
        help=(
            "Override data.partition_mode. shared keeps the old behavior; "
            "client_disjoint gives clients with the same dataset non-overlapping subsets."
        ),
    )
    parser.add_argument("--max-samples-per-client", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    args = parser.parse_args()

    load_dotenv()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    if args.data_partition_mode is not None:
        config.setdefault("data", {})["partition_mode"] = args.data_partition_mode

    if args.model_path is None:
        args.model_path = config.get("model", {}).get("model_id", DEFAULT_MODEL_PATH)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = load_backend(args.model_path, args.dtype)

    rows = []
    with output_path.open("w") as out:
        selected_mas = set(args.mas) if args.mas else None
        clients = [
            client
            for client in config["clients"]
            if selected_mas is None or client["mas"] in selected_mas
        ]
        if not clients:
            raise ValueError(f"No clients matched --mas {args.mas}")

        data_cfg = config.get("data", {})
        data_partition_mode = partition_mode(data_cfg)
        for client in clients:
            orchestrator = create_orchestrator(
                client["mas"],
                third_party_dir=args.third_party_dir,
                mode=args.orchestrator_mode,
            )
            dataset = load_dataset(client["dataset_name"])
            split = "validation" if "validation" in dataset else list(dataset.keys())[0]
            example_indices = client_partition_indices(
                len(dataset[split]),
                client,
                clients,
                int(config.get("seed", 42)),
                args.max_samples_per_client,
                data_partition_mode,
            )
            examples = dataset[split].select(example_indices)

            for idx, example in enumerate(examples):
                task_prompt, ground_truth = make_prompt(client, example)
                llm_generate = lambda prompt: generate(backend, prompt, args.max_new_tokens)
                start = time.perf_counter()
                orchestrated = orchestrator.run_inference(
                    client,
                    dict(example),
                    task_prompt,
                    llm_generate,
                )
                latency = time.perf_counter() - start
                response = orchestrated.metadata.get("response", "")
                prediction = normalize_prediction(client["dataset"], response)
                row = {
                    "client_id": client["id"],
                    "mas": client["mas"],
                    "dataset": client["dataset"],
                    "dataset_name": client["dataset_name"],
                    **client_partition_metadata(client, clients, data_partition_mode),
                    "split": split,
                    "example_index": idx,
                    "source_example_index": example_indices[idx],
                    "model_path": args.model_path,
                    "orchestrator_mode": args.orchestrator_mode,
                    "orchestrator_metadata": orchestrated.metadata,
                    "response": response,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "correct": is_correct(client["dataset"], prediction, ground_truth),
                    "latency_sec": round(latency, 4),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                rows.append(row)
                print(
                    f"{client['id']} {client['mas']} {client['dataset']}: "
                    f"pred={prediction!r} gt={ground_truth!r} "
                    f"correct={row['correct']} latency={row['latency_sec']}s"
                )

    correct = sum(row["correct"] for row in rows)
    print(f"wrote: {output_path}")
    print(f"accuracy: {correct}/{len(rows)}")


if __name__ == "__main__":
    main()
