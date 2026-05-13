from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import DatasetDict


def split_path(output_dir: str | Path, dataset: str, seed: int) -> Path:
    return Path(output_dir) / f"{dataset}_seed{seed}.json"


def ratio_split_indices(
    n_items: int,
    seed: int,
    train_ratio: float,
    eval_ratio: float,
) -> dict[str, list[int]]:
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)

    n_train = round(n_items * train_ratio)
    n_eval = round(n_items * eval_ratio)
    n_test = n_items - n_train - n_eval
    if n_test < 0:
        raise ValueError("train_ratio + eval_ratio must be <= 1.0")

    return {
        "train": indices[:n_train],
        "eval": indices[n_train : n_train + n_eval],
        "test": indices[n_train + n_eval :],
    }


def build_split_spec(
    dataset: str,
    dataset_name: str,
    dataset_dict: DatasetDict,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    dataset_cfg = config.get("datasets", {}).get(dataset, {})

    if dataset == "mmlu_pro":
        train_split = dataset_cfg.get("train_split", "validation")
        eval_split = dataset_cfg.get("eval_split", train_split)
        test_split = dataset_cfg.get("test_split", "test")
        return {
            "dataset": dataset,
            "dataset_name": dataset_name,
            "seed": seed,
            "splits": {
                "train": {
                    "source_split": train_split,
                    "indices": list(range(len(dataset_dict[train_split]))),
                },
                "eval": {
                    "source_split": eval_split,
                    "indices": list(range(len(dataset_dict[eval_split]))),
                },
                "test": {
                    "source_split": test_split,
                    "indices": list(range(len(dataset_dict[test_split]))),
                },
            },
        }

    if dataset == "gsm_hard":
        source_split = dataset_cfg.get("source_split", "train")
        train_ratio = float(dataset_cfg.get("train_ratio", 0.7))
        eval_ratio = float(dataset_cfg.get("eval_ratio", 0.15))
        indices = ratio_split_indices(
            len(dataset_dict[source_split]),
            seed,
            train_ratio,
            eval_ratio,
        )
        return {
            "dataset": dataset,
            "dataset_name": dataset_name,
            "seed": seed,
            "splits": {
                role: {"source_split": source_split, "indices": values}
                for role, values in indices.items()
            },
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def save_split_spec(spec: dict[str, Any], output_dir: str | Path) -> Path:
    path = split_path(output_dir, spec["dataset"], int(spec["seed"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    return path

