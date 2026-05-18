from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestrators.registry import adapter_statuses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedpao_six_clients.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    statuses = [status.__dict__ for status in adapter_statuses(config)]
    if args.json:
        print(json.dumps(statuses, indent=2, ensure_ascii=False))
        return

    for status in statuses:
        marker = "OK" if status["available"] and not status["missing"] else "MISSING"
        print(f"{status['name']:10s} {marker}")
        print(f"  repo: {status['repo_path']}")
        if status["missing"]:
            for path in status["missing"]:
                print(f"  missing: {path}")
        print(f"  note: {status['note']}")


if __name__ == "__main__":
    main()
