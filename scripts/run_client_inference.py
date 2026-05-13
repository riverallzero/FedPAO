import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

load_dotenv()

from src.data.loader import TaskExample, load_client_examples
from src.evaluation.metrics import score_prediction


DEFAULT_MODEL_PATH = "/data/digirl/checkpoints/Qwen3.5-2B-Base"


def load_causal_backend(model_path: str, torch_dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch_dtype,
        device_map="auto",
    )
    return {"kind": "causal_lm", "model": model.eval(), "tokenizer": tokenizer}


def load_image_text_backend(model_path: str, torch_dtype):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch_dtype,
        device_map="auto",
    )
    return {"kind": "image_text", "model": model.eval(), "processor": processor}


def load_backend(model_path: str, dtype: str, backend_kind: str):
    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    if backend_kind == "causal_lm":
        return load_causal_backend(model_path, torch_dtype)
    if backend_kind == "image_text":
        return load_image_text_backend(model_path, torch_dtype)

    try:
        return load_image_text_backend(model_path, torch_dtype)
    except Exception as image_text_error:
        backend = load_causal_backend(model_path, torch_dtype)
        backend["image_text_error"] = repr(image_text_error)
        return backend


def make_prompt(client: dict, example: TaskExample) -> str:
    mas = client["mas"]

    if example.dataset == "mmlu_pro":
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options = "\n".join(
            f"{letters[idx]}. {option}" for idx, option in enumerate(example.options or [])
        )
        return (
            f"You are the local frozen LLM backend for the {mas} client.\n"
            "Answer the multiple-choice question. Return only the option letter.\n\n"
            f"Question: {example.question}\n"
            f"Options:\n{options}\n"
            "Answer:"
        )

    if example.dataset == "gsm_hard":
        return (
            f"You are the local frozen LLM backend for the {mas} client.\n"
            "Solve the math problem. Return only the final numeric answer.\n\n"
            f"Problem: {example.question}\n"
            "Answer:"
        )

    raise ValueError(f"Unsupported dataset: {example.dataset}")


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


def generation_result(text: str, input_len: int, output_len: int) -> dict:
    generated_tokens = max(0, output_len - input_len)
    return {
        "text": text.strip(),
        "prompt_tokens": int(input_len),
        "generated_tokens": int(generated_tokens),
        "total_tokens": int(input_len + generated_tokens),
    }


def generate(backend: dict, prompt: str, max_new_tokens: int) -> dict:
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
        decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)
        return generation_result(decoded, input_len, output.shape[-1])

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
    decoded = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    return generation_result(decoded, input_len, output.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedpao_six_clients.yaml")
    parser.add_argument("--model-path")
    parser.add_argument("--output", default="outputs/client_inference_smoke.jsonl")
    parser.add_argument("--max-samples-per-client", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--split-role",
        choices=["train", "eval", "test"],
        default="eval",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--backend-kind",
        choices=["auto", "image_text", "causal_lm"],
        default=None,
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    if args.model_path is None:
        args.model_path = config.get("model", {}).get("model_id", DEFAULT_MODEL_PATH)
    if args.backend_kind is None:
        configured_backend = config.get("model", {}).get("backend", "hf_local")
        args.backend_kind = "causal_lm" if configured_backend == "hf_causal" else "auto"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = load_backend(args.model_path, args.dtype, args.backend_kind)

    rows = []
    with output_path.open("w") as out:
        for client in config["clients"]:
            examples = load_client_examples(
                config,
                client,
                args.split_role,
                args.max_samples_per_client,
            )

            for idx, example in enumerate(examples):
                prompt = make_prompt(client, example)
                start = time.perf_counter()
                response = generate(backend, prompt, args.max_new_tokens)
                latency = time.perf_counter() - start
                score = score_prediction(client["dataset"], response["text"], example.answer)
                row = {
                    "client_id": client["id"],
                    "mas": client["mas"],
                    "dataset": client["dataset"],
                    "dataset_name": client["dataset_name"],
                    "split_role": args.split_role,
                    "split": example.split,
                    "example_id": example.task_id,
                    "example_index": idx,
                    "model_path": args.model_path,
                    "response": response["text"],
                    "prediction": score["prediction"],
                    "ground_truth": score["ground_truth"],
                    "correct": score["correct"],
                    "accuracy": score["accuracy"],
                    "latency_sec": round(latency, 4),
                    "prompt_tokens": response["prompt_tokens"],
                    "generated_tokens": response["generated_tokens"],
                    "total_tokens": response["total_tokens"],
                    "agent_calls": 1,
                    "tool_calls": 0,
                    "redundant_call": 0,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                rows.append(row)
                print(
                    f"{client['id']} {client['mas']} {client['dataset']}: "
                    f"pred={row['prediction']!r} gt={row['ground_truth']!r} "
                    f"correct={row['correct']} latency={row['latency_sec']}s"
                )

    correct = sum(row["correct"] for row in rows)
    print(f"wrote: {output_path}")
    print(f"accuracy: {correct}/{len(rows)}")


if __name__ == "__main__":
    main()
