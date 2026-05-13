from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrators.registry import adapter_statuses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedpao_six_clients.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    rows = [status.__dict__ for status in adapter_statuses(config)]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    for row in rows:
        state = "OK" if row["available"] else "MISSING"
        print(f"{row['name']:10s} {state:8s} {row['repo_path']}")
        if row["missing"]:
            for missing in row["missing"]:
                print(f"  missing: {missing}")
        print(f"  note: {row['note']}")


if __name__ == "__main__":
    main()
