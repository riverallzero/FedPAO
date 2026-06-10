from __future__ import annotations

import random
from typing import Any


PARTITION_MODES = ("shared", "client_disjoint")


def partition_mode(data_cfg: dict[str, Any]) -> str:
    mode = str(data_cfg.get("partition_mode", "shared")).lower()
    if mode not in PARTITION_MODES:
        raise ValueError(
            f"Unsupported data.partition_mode={mode!r}. "
            f"Choose one of: {', '.join(PARTITION_MODES)}"
        )
    return mode


def dataset_partition_key(client: dict[str, Any]) -> str:
    return str(client.get("dataset_name") or client.get("dataset") or client["id"])


def shuffled_dataset_indices(
    total_size: int,
    client: dict[str, Any],
    seed: int,
) -> list[int]:
    indices = list(range(total_size))
    split_seed = seed + sum(ord(char) for char in dataset_partition_key(client))
    random.Random(split_seed).shuffle(indices)
    return indices


def client_dataset_peers(
    client: dict[str, Any],
    clients: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    key = dataset_partition_key(client)
    return [peer for peer in clients if dataset_partition_key(peer) == key]


def client_partition_indices(
    total_size: int,
    client: dict[str, Any],
    clients: list[dict[str, Any]],
    seed: int,
    sample_limit: int | None,
    mode: str,
) -> list[int]:
    indices = shuffled_dataset_indices(total_size, client, seed)
    if mode == "shared":
        return indices[:sample_limit] if sample_limit is not None else indices

    if mode != "client_disjoint":
        raise ValueError(
            f"Unsupported data.partition_mode={mode!r}. "
            f"Choose one of: {', '.join(PARTITION_MODES)}"
        )

    peers = client_dataset_peers(client, clients)
    peer_rank = next(
        index for index, peer in enumerate(peers) if peer["id"] == client["id"]
    )
    peer_count = len(peers)
    if sample_limit is None:
        return indices[peer_rank::peer_count]

    start = peer_rank * sample_limit
    end = start + sample_limit
    return indices[start:end]


def client_partition_metadata(
    client: dict[str, Any],
    clients: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    peers = client_dataset_peers(client, clients)
    peer_rank = next(
        index for index, peer in enumerate(peers) if peer["id"] == client["id"]
    )
    return {
        "partition_mode": mode,
        "partition_key": dataset_partition_key(client),
        "partition_rank": peer_rank,
        "partition_peers": len(peers),
    }
