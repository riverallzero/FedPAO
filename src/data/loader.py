from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from datasets import DatasetDict, load_dataset

from src.evaluation.normalization import normalize_answer


_DATASET_CACHE: dict[str, DatasetDict] = {}


@dataclass(frozen=True)
class TaskExample:
    task_id: str
    dataset: str
    split: str
    question: str
    options: list[str] | None
    answer: str
    raw: dict[str, Any]


def get_dataset(dataset_name: str) -> DatasetDict:
    if dataset_name not in _DATASET_CACHE:
        _DATASET_CACHE[dataset_name] = load_dataset(dataset_name)
    return _DATASET_CACHE[dataset_name]


def make_example_id(dataset: str, split: str, example: dict[str, Any]) -> str:
    payload = {
        "dataset": dataset,
        "split": split,
        "example": example,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def example_to_task(dataset: str, split: str, example: dict[str, Any]) -> TaskExample:
    if dataset == "mmlu_pro":
        return TaskExample(
            task_id=make_example_id(dataset, split, example),
            dataset=dataset,
            split=split,
            question=str(example["question"]),
            options=list(example["options"]),
            answer=normalize_answer(dataset, example["answer"]),
            raw=dict(example),
        )

    if dataset == "gsm_hard":
        return TaskExample(
            task_id=make_example_id(dataset, split, example),
            dataset=dataset,
            split=split,
            question=str(example["input"]),
            options=None,
            answer=normalize_answer(dataset, example["target"]),
            raw=dict(example),
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def configured_split(config: dict[str, Any], dataset: str, role: str) -> str | None:
    dataset_cfg = config.get("datasets", {}).get(dataset, {})
    if role == "train":
        return dataset_cfg.get("train_split") or dataset_cfg.get("source_split")
    if role == "eval":
        return dataset_cfg.get("eval_split") or dataset_cfg.get("validation_split")
    if role == "test":
        return dataset_cfg.get("test_split")
    if role == "source":
        return dataset_cfg.get("source_split")
    return None


def split_spec_path(config: dict[str, Any], dataset: str) -> Path:
    split_dir = config.get("data", {}).get("split_dir", "data/processed/splits")
    seed = int(config.get("seed", 42))
    return Path(split_dir) / f"{dataset}_seed{seed}.json"


def load_split_spec(config: dict[str, Any], dataset: str) -> dict[str, Any] | None:
    path = split_spec_path(config, dataset)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def default_split(dataset_dict: DatasetDict, role: str) -> str:
    if role == "eval" and "validation" in dataset_dict:
        return "validation"
    if role == "test" and "test" in dataset_dict:
        return "test"
    if role == "train" and "train" in dataset_dict:
        return "train"
    return list(dataset_dict.keys())[0]


def load_client_examples(
    config: dict[str, Any],
    client: dict[str, Any],
    role: str,
    max_samples: int | None = None,
) -> list[TaskExample]:
    dataset_dict = get_dataset(client["dataset_name"])
    split_spec = load_split_spec(config, client["dataset"])
    if split_spec and role in split_spec.get("splits", {}):
        role_spec = split_spec["splits"][role]
        split = role_spec["source_split"]
        indices = role_spec["indices"]
        if split not in dataset_dict:
            raise KeyError(
                f"Dataset {client['dataset_name']} has no split {split!r}; "
                f"available splits: {list(dataset_dict.keys())}"
            )
        if max_samples is not None:
            indices = indices[:max_samples]
        return [
            example_to_task(client["dataset"], split, dataset_dict[split][idx])
            for idx in indices
        ]

    split = configured_split(config, client["dataset"], role) or default_split(
        dataset_dict, role
    )
    if split not in dataset_dict:
        raise KeyError(
            f"Dataset {client['dataset_name']} has no split {split!r}; "
            f"available splits: {list(dataset_dict.keys())}"
        )

    source = dataset_dict[split]
    limit = len(source) if max_samples is None else min(max_samples, len(source))
    return [
        example_to_task(client["dataset"], split, source[idx])
        for idx in range(limit)
    ]
